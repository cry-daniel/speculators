import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

SCRIPTS_DIR = (
    Path(__file__).resolve().parent
    / "examples"
    / "evaluate"
    / "eval-guidellm"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import offline_24_pruning  # noqa: E402
import aggregate_lm_eval_accuracy  # noqa: E402
import prepare_covwanda_gate_masks  # noqa: E402
import run_lm_eval_accuracy  # noqa: E402
import run_speclink_system_matrix  # noqa: E402
import run_structured_24_spec_quality  # noqa: E402
import run_token_dense_accuracy  # noqa: E402
import token_dense_methods  # noqa: E402
from vllm import speclink_linear  # noqa: E402
from vllm import speclink_mlp  # noqa: E402
from vllm import speclink_structured_24  # noqa: E402
from vllm import speclink_token_dense  # noqa: E402
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    prepare_cutlass_sparse24_gate_up_swiglu,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_grouped_owner_linear_prepacked,
    sparse24_cutlass_grouped_owner_swiglu_prepacked,
    sparse24_cutlass_residual_delta_swiglu_prepacked,
    sparse24_cutlass_routed_exact_linear_prepacked,
    sparse24_cutlass_routed_exact_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_routed_swiglu_delta_,
)


class TokenDenseSparse24Tests(unittest.TestCase):
    def tearDown(self) -> None:
        speclink_token_dense._pending_scores.clear()
        speclink_token_dense._graph_capture_plans.clear()
        speclink_token_dense._graph_plan_buffers.clear()
        speclink_token_dense._batch_route_credit = 0.0
        speclink_token_dense._batch_route_remaining = 0
        speclink_token_dense._batch_route_sparse = False

    def test_verify_mask_uses_topk_prefix_product_confidence(self) -> None:
        speclink_token_dense._pending_scores["req0"].append([0.99] * 20)
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "16",
                },
            ):
                mask = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["req0"],
                    num_scheduled_tokens=[22],
                    num_draft_tokens=[20],
                    num_decode_draft_tokens=[20],
                    cu_num_scheduled_tokens=[22],
                    total_num_scheduled_tokens=22,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(mask)
        assert mask is not None
        expected = [True] * 22
        expected[17:21] = [False] * 4
        self.assertEqual(mask.dense_mask.tolist(), expected)
        self.assertEqual(mask.dense_count, 18)
        self.assertEqual(mask.sparse_count, 4)
        self.assertEqual(
            mask.sparse_rows[: mask.sparse_count].tolist(),
            [17, 18, 19, 20],
        )

    def test_accuracy_delta_uses_dense_eagle3_baseline(self) -> None:
        rows = [
            {
                "model_label": "qwen3_8b",
                "task_result_name": "gsm8k_cot",
                "mode": "dense_ar",
                "score": 0.95,
                "request_output_tokens_per_second": 80.0,
            },
            {
                "model_label": "qwen3_8b",
                "task_result_name": "gsm8k_cot",
                "mode": "eagle3_dense",
                "score": 0.90,
                "request_output_tokens_per_second": 100.0,
            },
            {
                "model_label": "qwen3_8b",
                "task_result_name": "gsm8k_cot",
                "mode": "token_dense_d8",
                "score": 0.86,
                "request_output_tokens_per_second": 131.0,
            },
        ]

        aggregate_lm_eval_accuracy.add_accuracy_comparisons(rows)
        aggregate_lm_eval_accuracy.add_throughput_comparisons(rows)
        aggregate_lm_eval_accuracy.add_goal_checks(rows)

        self.assertEqual(rows[2]["eagle3_dense_score"], 0.90)
        self.assertAlmostEqual(rows[2]["delta_pp_vs_eagle3_dense"], -4.0)
        self.assertTrue(rows[2]["accuracy_within_5pp"])
        self.assertTrue(rows[2]["meets_hard_goal"])
        self.assertFalse(rows[2]["meets_target_goal"])

    def test_lm_eval_server_warmup_uses_serving_batch_size(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        args = SimpleNamespace(
            server_warmup_batches=2,
            server_warmup_max_tokens=16,
            num_concurrent=32,
            request_timeout_s=30,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            with mock.patch.object(
                run_lm_eval_accuracy.request,
                "urlopen",
                return_value=FakeResponse(),
            ) as urlopen:
                run_lm_eval_accuracy.warmup_server(
                    args,
                    port=8123,
                    model_path="model-under-test",
                    run_dir=run_dir,
                )

            self.assertEqual(urlopen.call_count, 2)
            sent = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(sent["model"], "model-under-test")
            self.assertEqual(len(sent["prompt"]), 32)
            self.assertEqual(sent["max_tokens"], 16)
            record = json.loads(
                (run_dir / "server_warmup.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["completed_batches"], 2)
            self.assertEqual(record["prompts_per_batch"], 32)

    def test_budget_search_accepts_d8_and_dense_control(self) -> None:
        self.assertEqual(
            token_dense_methods.parse_method_config(
                "token_dense_d8"
            ).token_dense_budget,
            8,
        )
        self.assertEqual(
            token_dense_methods.parse_method_config(
                "token_dense_d256"
            ).token_dense_budget,
            256,
        )

    def test_dynamic_budget_scales_with_scored_rows_and_active_requests(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SPECLINK_TOKEN_DENSE_BUDGET_MODE": "dynamic",
                "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.25",
                "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST": "1",
                "SPECLINK_TOKEN_DENSE_DENSE_CAP": "18",
            },
        ):
            self.assertEqual(
                speclink_token_dense.effective_dense_token_budget(80, 16),
                18,
            )
            self.assertEqual(
                speclink_token_dense.effective_dense_token_budget(12, 16),
                12,
            )

    def test_dynamic_budget_rejects_cap_below_request_floor(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SPECLINK_TOKEN_DENSE_BUDGET_MODE": "dynamic",
                "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.25",
                "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST": "1",
                "SPECLINK_TOKEN_DENSE_DENSE_CAP": "8",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "per-request floor"):
                speclink_token_dense.effective_dense_token_budget(80, 16)

    def test_dynamic_method_exports_budget_formula_and_selection(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        args.token_dense_dense_ratio = 0.125
        args.token_dense_dense_min_per_request = 1
        args.token_dense_dense_cap = 64
        args.token_dense_dense_selection = "balanced_prefix"
        args.token_dense_balanced_start_position = 1
        args.token_dense_mixed_layers = "4-7"
        args.token_dense_mlp_static_layers = "5-6"
        args.token_dense_o_sparse_layers = "3,11-25"
        args.token_dense_mixed_projection_policy = "mlp"
        args.token_dense_sparse_unscored_decode = True
        method = token_dense_methods.parse_method_config("token_dense_dynamic")

        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )

        self.assertEqual(method.token_dense_budget_mode, "dynamic")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_BUDGET_MODE"], "dynamic")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_DENSE_RATIO"], "0.125")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST"], "1")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_DENSE_CAP"], "64")
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_DENSE_SELECTION"],
            "balanced_prefix",
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION"],
            "1",
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_MIXED_LAYERS"], "4-7")
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS"],
            "5-6",
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_O_SPARSE_LAYERS"],
            "3,11-25",
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY"],
            "mlp",
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE"],
            "1",
        )

    def test_system_matrix_forwards_sparse_unscored_decode(self) -> None:
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["sparse_unscored_decode"] = True

        command = run_speclink_system_matrix.candidate_args(config)

        self.assertIn("--token-dense-sparse-unscored-decode", command)

    def test_system_matrix_forwards_o_sparse_layers(self) -> None:
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["o_sparse_layers"] = "3,11-25"

        command = run_speclink_system_matrix.candidate_args(config)

        option = command.index("--token-dense-o-sparse-layers")
        self.assertEqual(command[option + 1], "3,11-25")

    def test_system_matrix_forwards_mlp_static_layers(self) -> None:
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["mlp_static_layers"] = "5-6,21-22,29"

        command = run_speclink_system_matrix.candidate_args(config)

        option = command.index("--token-dense-mlp-static-layers")
        self.assertEqual(command[option + 1], "5-6,21-22,29")

    def test_inline_swiglu_option_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            [
                "--token-dense-fused-batch-mlp",
                "--token-dense-inline-swiglu-mlp",
            ]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP"], "1")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_INLINE_SWIGLU_MLP"], "1")

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["inline_swiglu_mlp"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-inline-swiglu-mlp", command)

    def test_direct_store_gate_up_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-direct-store-gate-up"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_SPARSE24_DIRECT_STORE_GATE_UP"], "1")

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["direct_store_gate_up"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-direct-store-gate-up", command)

    def test_qkv_heterogeneous_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            [
                "--token-dense-linear-strategy",
                "full_sparse_residual",
                "--token-dense-qkv-heterogeneous-routing",
                "--token-dense-qkv-heterogeneous-max-rows",
                "176",
            ]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_QKV_HETEROGENEOUS_ROUTING"],
            "1",
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_QKV_HETEROGENEOUS_MAX_ROWS"],
            "176",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["linear_strategy"] = "full_sparse_residual"
        config["qkv_heterogeneous_routing"] = True
        config["qkv_heterogeneous_max_rows"] = 176
        command = run_speclink_system_matrix.candidate_args(config)
        option_index = command.index(
            "--token-dense-qkv-heterogeneous-routing"
        )
        self.assertEqual(
            command[option_index + 1 : option_index + 3],
            ["--token-dense-qkv-heterogeneous-max-rows", "176"],
        )

    def test_fused_override_options_reach_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            [
                "--token-dense-full-sparse-override-mlp",
                "--token-dense-full-sparse-override-mlp-min-rows",
                "288",
                "--token-dense-qkv-paired-routing",
                "--token-dense-qkv-paired-max-rows",
                "576",
                "--token-dense-qkv-active-wave-c12",
            ]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP"], "1"
        )
        self.assertEqual(
            env[
                "SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP_MIN_ROWS"
            ],
            "288",
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_QKV_PAIRED_ROUTING"], "1"
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_QKV_PAIRED_MAX_ROWS"], "576"
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_QKV_ACTIVE_WAVE_C12"], "1"
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["full_sparse_override_mlp"] = True
        config["full_sparse_override_mlp_min_rows"] = 288
        config["qkv_paired_routing"] = True
        config["qkv_paired_max_rows"] = 576
        config["qkv_active_wave_c12"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-full-sparse-override-mlp", command)
        self.assertIn("--token-dense-qkv-paired-routing", command)
        self.assertIn("--token-dense-qkv-active-wave-c12", command)
        self.assertIn("288", command)
        self.assertIn("576", command)

    def test_qkv_heterogeneous_config_matches_sparse_accumulator(self) -> None:
        with mock.patch.object(
            speclink_linear,
            "_SPARSE24_ACCUMULATOR",
            "fp16_qkv_gate",
        ):
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(112, 16),
                "256x32x64_s3_sw4_f16",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(224, 32),
                "256x64_sparse_128x32_dense_s3_f16",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(288, 32),
                "256x64_sparse_128x32_dense_s3_f16",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(448, 64),
                "256x128_sparse_128x64_dense_s2_f16",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(576, 64),
                "256x64_sparse_128x64_dense_s3_f16",
            )
            high_budget_expected = {
                (112, 30): "256x32x64_s3_sw4_f16",
                (144, 40): "256x64_sparse_128x32_dense_s3_f16",
                (176, 50): "256x64_sparse_128x32_dense_s3_f16",
                (224, 60): "256x64_sparse_128x32_dense_s3_f16",
                (288, 80): "256x128_sparse_128x64_dense_s2_f16",
                (352, 100): "256x128_sparse_128x64_dense_s2_f16",
                (448, 120): "256x128_sparse_128x64_dense_s2_f16",
                (576, 160): "256x64_sparse_128x64_dense_s3_f16",
                (704, 200): "256x128_sparse_128x64_dense_s2_f16",
            }
            for (rows, dense_count), expected in high_budget_expected.items():
                with self.subTest(rows=rows, dense_count=dense_count):
                    self.assertEqual(
                        speclink_linear._qkv_heterogeneous_config(
                            rows,
                            dense_count,
                            6144,
                        ),
                        expected,
                    )
        with mock.patch.object(
            speclink_linear,
            "_SPARSE24_ACCUMULATOR",
            "fp32",
        ):
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(112, 16),
                "256x32x64_s3_sw4",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(224, 32),
                "256x64_sparse_128x32_dense_s3",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(288, 32),
                "256x64_sparse_128x32_dense_s3",
            )
            self.assertEqual(
                speclink_linear._qkv_heterogeneous_config(448, 64),
                "256x128_sparse_128x64_dense_s2",
            )

    def test_qkv_heterogeneous_exact_dispatches_high_dense_budget(self) -> None:
        input_tensor = mock.Mock()
        input_tensor.shape = (288, 4096)
        input_tensor.is_cuda = True
        input_tensor.dtype = torch.float16
        input_tensor.device = torch.device("cuda")
        input_tensor.is_contiguous.return_value = True
        dense_weight = mock.Mock()
        dense_weight.shape = (6144, 4096)
        dense_weight.dtype = torch.float16
        dense_weight.is_contiguous.return_value = True
        dense_rows = mock.Mock()
        dense_rows.numel.return_value = 80
        sparse_rows = mock.Mock()
        sparse_rows.numel.return_value = 208
        expected = mock.Mock()

        with (
            mock.patch.object(
                speclink_linear,
                "_QKV_HETEROGENEOUS_ROUTING_ENABLED",
                True,
            ),
            mock.patch.object(
                speclink_linear,
                "_QKV_HETEROGENEOUS_MAX_ROWS",
                704,
            ),
            mock.patch.object(
                speclink_linear,
                "_SPARSE24_ACCUMULATOR",
                "fp16_qkv_gate",
            ),
            mock.patch.object(
                speclink_linear,
                "_reuse_sparse_buffers_enabled",
                return_value=False,
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_heterogeneous_linear_prepacked",
                return_value=expected,
            ) as heterogeneous,
        ):
            actual = speclink_linear._qkv_heterogeneous_exact(
                input_tensor,
                dense_weight,
                "values",
                "meta",
                dense_rows,
                sparse_rows,
            )

        self.assertIs(actual, expected)
        self.assertEqual(
            heterogeneous.call_args.kwargs["config"],
            "256x128_sparse_128x64_dense_s2_f16",
        )

    def test_cutlass_down_fp16_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-cutlass-down-fp16"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_CUTLASS_DOWN_FP16"], "1")

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["cutlass_down_fp16"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-cutlass-down-fp16", command)

    def test_cutlass_down_fp16_selector_uses_benchmarked_shapes(self) -> None:
        self.assertEqual(
            speclink_mlp._cutlass_down_fp16_config(112), "64x64x64_s4"
        )
        self.assertEqual(
            speclink_mlp._cutlass_down_fp16_config(288), "64x128x64_s3"
        )
        self.assertEqual(
            speclink_mlp._cutlass_down_fp16_config(352), "128x128x64_s3"
        )
        self.assertEqual(
            speclink_mlp._cutlass_down_fp16_config(704), "128x128x64_s3"
        )
        self.assertIsNone(speclink_mlp._cutlass_down_fp16_config(104))

    def test_routed_swiglu_option_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            [
                "--token-dense-fused-batch-mlp",
                "--token-dense-linear-strategy",
                "full_sparse_residual",
                "--token-dense-mlp-strategy",
                "linear",
                "--token-dense-routed-swiglu-mlp",
            ]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP"], "1")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_ROUTED_SWIGLU_MLP"], "1")

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["routed_swiglu_mlp"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-routed-swiglu-mlp", command)

    def test_indexed_down_option_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-indexed-down-epilogue"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_INDEXED_DOWN_EPILOGUE"],
            "1",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["indexed_down_epilogue"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-indexed-down-epilogue", command)

    def test_uninitialized_workspace_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--no-token-dense-uninitialized-routed-workspace"]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_UNINITIALIZED_ROUTED_WORKSPACE"],
            "0",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["uninitialized_routed_workspace"] = False
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn(
            "--no-token-dense-uninitialized-routed-workspace",
            command,
        )

    def test_paired_persistent_gate_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--no-token-dense-paired-persistent-gate"]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_GATE"],
            "0",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["paired_persistent_gate"] = False
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn(
            "--no-token-dense-paired-persistent-gate",
            command,
        )

    def test_paired_persistent_gate_selector_uses_benchmarked_shapes(
        self,
    ) -> None:
        with mock.patch.object(
            speclink_mlp,
            "_PAIRED_PERSISTENT_GATE_ENABLED",
            True,
        ):
            self.assertTrue(
                speclink_mlp._use_paired_persistent_gate(224, 56, 24576)
            )
            self.assertFalse(
                speclink_mlp._use_paired_persistent_gate(144, 18, 24576)
            )
            self.assertTrue(
                speclink_mlp._use_paired_persistent_gate(144, 32, 24576)
            )
            self.assertTrue(
                speclink_mlp._use_paired_persistent_gate(112, 28, 28672)
            )
            self.assertFalse(
                speclink_mlp._use_paired_persistent_gate(224, 56, 1024)
            )

    def test_paired_persistent_down_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--no-token-dense-paired-persistent-down"]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_DOWN"],
            "0",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["paired_persistent_down"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn(
            "--token-dense-paired-persistent-down",
            command,
        )

    def test_paired_persistent_down_selector_uses_positive_shapes(self) -> None:
        with mock.patch.object(
            speclink_mlp,
            "_PAIRED_PERSISTENT_DOWN_ENABLED",
            True,
        ):
            self.assertTrue(
                speclink_mlp._use_paired_persistent_down(448, 112, 12288)
            )
            self.assertFalse(
                speclink_mlp._use_paired_persistent_down(576, 128, 12288)
            )
            self.assertTrue(
                speclink_mlp._use_paired_persistent_down(288, 64, 14336)
            )
            self.assertFalse(
                speclink_mlp._use_paired_persistent_down(224, 56, 14336)
            )

    def test_paired_gather_down_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-paired-gather-down"]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_PAIRED_GATHER_DOWN"],
            "1",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["paired_gather_down"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-paired-gather-down", command)

    def test_paired_gather_down_selector_uses_profiled_shapes(self) -> None:
        with mock.patch.object(
            speclink_mlp,
            "_PAIRED_GATHER_DOWN_ENABLED",
            True,
        ):
            self.assertEqual(
                speclink_mlp._paired_gather_down_config(288, 32, 12288),
                "256x32_full_256x32_residual_contiguous",
            )
            self.assertEqual(
                speclink_mlp._paired_gather_down_config(448, 32, 12288),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertEqual(
                speclink_mlp._paired_gather_down_config(144, 40, 14336),
                "256x32_full_256x32_residual_contiguous",
            )
            self.assertEqual(
                speclink_mlp._paired_gather_down_config(576, 64, 14336),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertIsNone(
                speclink_mlp._paired_gather_down_config(704, 32, 12288)
            )
            self.assertIsNone(
                speclink_mlp._paired_gather_down_config(288, 64, 14336)
            )

    def test_sparse_gate_dense_down_option_reaches_server_env_and_matrix(
        self,
    ) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-sparse-gate-dense-down"]
        )
        method = token_dense_methods.parse_method_config(
            "token_dense_dynamic"
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_SPARSE_GATE_DENSE_DOWN"],
            "1",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["sparse_gate_dense_down"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-sparse-gate-dense-down", command)

    def test_sparse_gate_dense_down_avoids_swiglu_graph_shape_cliff(
        self,
    ) -> None:
        for rows in (72, 80, 88, 96):
            self.assertEqual(
                speclink_mlp._sparse_gate_dense_down_config(rows, 24576),
                "256x32x64_s3_sw4",
            )
            self.assertEqual(
                speclink_mlp._sparse_gate_dense_down_config(rows, 28672),
                "256x32x64_s3_sw4",
            )
        self.assertEqual(
            speclink_mlp._sparse_gate_dense_down_config(104, 24576),
            "auto",
        )
        with mock.patch.object(
            speclink_mlp,
            "_SPARSE_GATE_SWIGLU_FP16_ENABLED",
            True,
        ):
            for rows in (112, 144, 176, 224, 288, 352, 448, 576, 704):
                self.assertEqual(
                    speclink_mlp._sparse_gate_dense_down_config(rows, 24576),
                    "256x64x64_s3_sw4_f16",
                )
                self.assertEqual(
                    speclink_mlp._sparse_gate_dense_down_config(rows, 28672),
                    "256x64x64_s3_sw4_f16",
                )

    def test_indexed_output_option_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-indexed-output-epilogue"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_INDEXED_OUTPUT_EPILOGUE"],
            "1",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["indexed_output_epilogue"] = True
        command = run_speclink_system_matrix.candidate_args(config)
        self.assertIn("--token-dense-indexed-output-epilogue", command)


    def test_system_matrix_forwards_explicit_mask_root(self) -> None:
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["mask_root"] = "/tmp/composed_masks"

        command = run_speclink_system_matrix.candidate_args(config)

        root_index = command.index("--token-dense-mask-root") + 1
        self.assertEqual(command[root_index], "/tmp/composed_masks")

    def test_retain_dense_weight_policy_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--token-dense-retain-dense-weight", "qkv"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_SPARSE24_RETAIN_DENSE_WEIGHT"], "qkv")

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["retain_dense_weight"] = "qkv"
        command = run_speclink_system_matrix.candidate_args(config)
        policy_index = command.index("--token-dense-retain-dense-weight") + 1
        self.assertEqual(command[policy_index], "qkv")

    def test_attention_backend_is_common_to_dense_and_speclink(self) -> None:
        args = run_speclink_system_matrix.build_parser().parse_args(
            ["--attention-backend", "FLASHINFER"]
        )
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        commands = []
        for variant in ("dense", "speclink"):
            case = run_speclink_system_matrix.RunCase(
                phase="performance",
                model="qwen3_8b",
                task="math_reasoning",
                batch_size=64,
                k=6,
                repeat=1,
                variant=variant,
            )
            commands.append(
                run_speclink_system_matrix.build_command(
                    args,
                    case,
                    Path("/tmp/run"),
                    config,
                )
            )
        for command in commands:
            backend_index = command.index("--attention-backend") + 1
            self.assertEqual(command[backend_index], "FLASHINFER")

    def test_parallel_mixed_override_reaches_server_env_and_matrix(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args(
            ["--no-token-dense-parallel-mixed-override"]
        )
        method = token_dense_methods.parse_method_config("token_dense_dynamic")

        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_PARALLEL_MIXED_OVERRIDE"],
            "0",
        )

        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        config["parallel_mixed_override"] = False
        self.assertIn(
            "--no-token-dense-parallel-mixed-override",
            run_speclink_system_matrix.candidate_args(config),
        )

    def test_comparisons_reject_mismatched_k_or_batch_size(self) -> None:
        rows = [
            {
                "model_label": "qwen3_8b",
                "task_result_name": "math_reasoning",
                "mode": "eagle3_dense",
                "num_spec_tokens": 8,
                "batch_size": 32,
                "request_output_tokens_per_second": 100.0,
            },
            {
                "model_label": "qwen3_8b",
                "task_result_name": "math_reasoning",
                "mode": "token_dense_dynamic",
                "num_spec_tokens": 6,
                "batch_size": 32,
                "request_output_tokens_per_second": 150.0,
            },
            {
                "model_label": "qwen3_8b",
                "task_result_name": "math_reasoning",
                "mode": "token_dense_dynamic",
                "num_spec_tokens": 8,
                "batch_size": 32,
                "request_output_tokens_per_second": 141.0,
            },
        ]

        aggregate_lm_eval_accuracy.add_throughput_comparisons(rows)

        self.assertEqual(rows[1]["speedup_vs_eagle3_dense"], "")
        self.assertAlmostEqual(rows[2]["speedup_vs_eagle3_dense"], 1.41)

    def test_system_matrix_pairs_same_k_and_requires_all_repeats(self) -> None:
        parser = run_speclink_system_matrix.build_parser()
        args = parser.parse_args(
            [
                "--phase",
                "performance",
                "--models",
                "qwen3_8b",
                "--tasks",
                "math_reasoning",
                "--batch-sizes",
                "32",
                "--k-values",
                "8",
                "--performance-repeats",
                "3",
            ]
        )
        cases = run_speclink_system_matrix.build_cases(args)

        self.assertEqual(len(cases), 6)
        self.assertEqual({case.k for case in cases}, {8})
        for repeat in (1, 2, 3):
            variants = {
                case.variant for case in cases if case.repeat == repeat
            }
            self.assertEqual(variants, {"dense", "speclink"})

        rows = {
            case.key: {
                "phase": case.phase,
                "model": case.model,
                "task": case.task,
                "batch_size": case.batch_size,
                "k": case.k,
                "repeat": case.repeat,
                "variant": case.variant,
                "service_config_id": "same-service",
                "request_output_tokens_per_second": (
                    100.0 if case.variant == "dense" else 141.0
                ),
            }
            for case in cases[:4]
        }
        comparisons = run_speclink_system_matrix.build_comparisons(
            rows,
            performance_repeats=3,
        )
        self.assertFalse(comparisons[0]["complete"])
        self.assertTrue(comparisons[0]["service_config_match"])
        self.assertAlmostEqual(comparisons[0]["speedup_median"], 1.41)

    def test_system_matrix_rejects_mismatched_service_config(self) -> None:
        rows = {}
        for variant, service_id, rate in (
            ("dense", "full_decode_only", 100.0),
            ("speclink", "full_and_piecewise", 150.0),
        ):
            key = (
                "performance",
                "qwen3_8b",
                "gsm8k_cot",
                64,
                6,
                1,
                variant,
            )
            rows[key] = {
                "phase": "performance",
                "model": "qwen3_8b",
                "task": "gsm8k_cot",
                "batch_size": 64,
                "k": 6,
                "repeat": 1,
                "variant": variant,
                "service_config_id": service_id,
                "status": "ok",
                "request_output_tokens_per_second": rate,
            }

        comparison = run_speclink_system_matrix.build_comparisons(
            rows,
            performance_repeats=1,
        )[0]

        self.assertFalse(comparison["service_config_match"])
        self.assertFalse(comparison["complete"])
        self.assertIsNone(comparison["speedup_median"])

    def test_system_matrix_service_config_tracks_graph_mode(self) -> None:
        args = run_speclink_system_matrix.build_parser().parse_args(
            [
                "--phase",
                "performance",
                "--models",
                "qwen3_8b",
                "--tasks",
                "gsm8k_cot",
                "--batch-sizes",
                "64",
                "--k-values",
                "6",
            ]
        )
        case = run_speclink_system_matrix.build_cases(args)[0]
        config = dict(run_speclink_system_matrix.DEFAULT_CANDIDATE_CONFIG)
        full_decode_id = run_speclink_system_matrix.service_config_id(
            args,
            case,
            config,
            "runtime",
        )
        config["cudagraph_mode"] = "full_and_piecewise"
        full_graph_id = run_speclink_system_matrix.service_config_id(
            args,
            case,
            config,
            "runtime",
        )

        self.assertNotEqual(full_decode_id, full_graph_id)

    def test_system_matrix_accuracy_uses_matching_batch_sizes(self) -> None:
        args = run_speclink_system_matrix.build_parser().parse_args(
            [
                "--phase",
                "accuracy",
                "--models",
                "qwen3_8b",
                "--tasks",
                "gsm8k_cot",
                "--batch-sizes",
                "16,32",
                "--k-values",
                "6,8",
            ]
        )
        cases = run_speclink_system_matrix.build_cases(args)

        self.assertEqual(len(cases), 8)
        self.assertEqual({case.batch_size for case in cases}, {16, 32})
        config = run_speclink_system_matrix.load_candidate_configs(
            None,
            ["qwen3_8b"],
        )["qwen3_8b"]
        command = run_speclink_system_matrix.build_command(
            args,
            cases[0],
            Path("temp/run"),
            config,
        )
        concurrent_index = command.index("--num-concurrent") + 1
        self.assertEqual(int(command[concurrent_index]), cases[0].batch_size)

    def test_external_baseline_does_not_replace_local_baseline(self) -> None:
        local = [
            {
                "model_label": "qwen3_8b",
                "task_result_name": "gsm8k_cot",
                "mode": "eagle3_dense",
                "score": 0.9,
            }
        ]
        with mock.patch.object(
            aggregate_lm_eval_accuracy,
            "rows_from_runs",
            return_value=[
                {
                    "model_label": "qwen3_8b",
                    "task_result_name": "gsm8k_cot",
                    "mode": "eagle3_dense",
                    "score": 0.8,
                },
                {
                    "model_label": "llama3_1_8b",
                    "task_result_name": "gsm8k_cot",
                    "mode": "eagle3_dense",
                    "score": 0.7,
                },
            ],
        ):
            aggregate_lm_eval_accuracy.add_external_baselines(
                local,
                Path("/tmp/baseline"),
            )

        self.assertEqual(len(local), 2)
        self.assertEqual(local[0]["score"], 0.9)
        self.assertEqual(local[1]["model_label"], "llama3_1_8b")

    def test_lm_eval_rows_include_shared_sparse_scale_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "token_dense_sparse_value_scale": 1.05,
                        "token_dense_row_scale_mode": "variance",
                        "token_dense_row_scale_max": 1.5,
                        "token_dense_sparse_output_mode": "view_mlp_o",
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "qwen3_8b" / "token_dense_d0" / "gsm8k_cot"
            run_dir.mkdir(parents=True)
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "model_label": "qwen3_8b",
                        "mode": "token_dense_d0",
                        "task": "gsm8k_cot",
                        "status": "failed",
                    }
                ),
                encoding="utf-8",
            )

            rows = aggregate_lm_eval_accuracy.rows_from_runs(root)

        self.assertEqual(rows[0]["token_dense_sparse_value_scale"], 1.05)
        self.assertEqual(rows[0]["token_dense_row_scale_mode"], "variance")
        self.assertEqual(rows[0]["token_dense_row_scale_max"], 1.5)
        self.assertEqual(
            rows[0]["token_dense_sparse_output_mode"],
            "view_mlp_o",
        )

    def test_projection_policy_selects_only_requested_target_modules(self) -> None:
        enabled = speclink_structured_24._token_dense_projection_enabled

        self.assertFalse(enabled("none", "qkv_proj"))
        self.assertFalse(enabled("none", "gate_up_proj"))
        self.assertTrue(enabled("all", "qkv_proj"))
        self.assertTrue(enabled("mlp", "gate_up_proj"))
        self.assertTrue(enabled("mlp", "down_proj"))
        self.assertFalse(enabled("mlp", "o_proj"))
        self.assertTrue(enabled("attention", "qkv_proj"))
        self.assertTrue(enabled("attention", "o_proj"))
        self.assertFalse(enabled("attention", "gate_up_proj"))
        self.assertTrue(enabled("gate_up", "gate_up_proj"))
        self.assertFalse(enabled("gate_up", "down_proj"))
        self.assertTrue(enabled("o_gate_up", "o_proj"))
        self.assertTrue(enabled("o_gate_up", "gate_up_proj"))
        self.assertFalse(enabled("o_gate_up", "qkv_proj"))
        self.assertFalse(enabled("o_gate_up", "down_proj"))
        self.assertTrue(enabled("qkv_gate_up_down", "qkv_proj"))
        self.assertTrue(enabled("qkv_gate_up_down", "gate_up_proj"))
        self.assertTrue(enabled("qkv_gate_up_down", "down_proj"))
        self.assertFalse(enabled("qkv_gate_up_down", "o_proj"))
        self.assertTrue(enabled("attention_gate_up", "qkv_proj"))
        self.assertTrue(enabled("attention_gate_up", "o_proj"))
        self.assertTrue(enabled("attention_gate_up", "gate_up_proj"))
        self.assertFalse(enabled("attention_gate_up", "down_proj"))
        with self.assertRaisesRegex(RuntimeError, "PROJECTION_POLICY"):
            enabled("unknown", "qkv_proj")

    def test_precomputed_log_scores_preserve_topk_routing(self) -> None:
        prefix_log_scores = torch.arange(1, 21, dtype=torch.float32) * torch.log(
            torch.tensor(0.99)
        )
        speclink_token_dense._pending_scores["req0"].append(
            speclink_token_dense._CumulativeLogScores(prefix_log_scores)
        )
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "16",
                    "SPECLINK_TOKEN_DENSE_FAST_PLAN": "1",
                },
            ):
                mask = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["req0"],
                    num_scheduled_tokens=[22],
                    num_draft_tokens=[20],
                    num_decode_draft_tokens=[20],
                    cu_num_scheduled_tokens=[22],
                    total_num_scheduled_tokens=22,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(mask)
        assert mask is not None
        expected = [True] * 22
        expected[17:21] = [False] * 4
        self.assertEqual(mask.dense_mask.tolist(), expected)
        self.assertEqual(mask.dense_count, 18)
        self.assertEqual(mask.sparse_count, 4)

    def test_verify_mask_keeps_bonus_logit_dense(self) -> None:
        speclink_token_dense._pending_scores["req0"].append([0.99] * 8)
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0",
                },
            ):
                plan = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["req0"],
                    num_scheduled_tokens=[9],
                    num_draft_tokens=[8],
                    num_decode_draft_tokens=[8],
                    cu_num_scheduled_tokens=[9],
                    total_num_scheduled_tokens=9,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.dense_mask.tolist(), [False] * 8 + [True])
        self.assertEqual(plan.sparse_rows[:8].tolist(), list(range(8)))

    def test_sparse_bonus_trades_bonus_for_second_prefix_row(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "4",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "balanced_prefix",
                    "SPECLINK_TOKEN_DENSE_SPARSE_BONUS": "1",
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {4},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, True, False, False, False] * 2,
        )
        self.assertEqual(plan.dense_count, 4)
        self.assertEqual(plan.sparse_count, 6)

    def test_balanced_prefix_budget_protects_each_request(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "2",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "balanced_prefix",
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {2},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )
            self.assertFalse(speclink_token_dense.draft_scores_required())

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, False, False, False, True] * 2,
        )
        self.assertEqual(plan.dense_rows[: plan.dense_count].tolist(), [0, 4, 5, 9])
        self.assertEqual(
            plan.sparse_rows[: plan.sparse_count].tolist(),
            [1, 2, 3, 6, 7, 8],
        )

    def test_balanced_budget_can_start_from_second_position(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "2",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "balanced_prefix",
                    "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION": "1",
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {2},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [False, True, False, False, True] * 2,
        )

    def test_balanced_confidence_protects_each_request_then_ranks_suffix(
        self,
    ) -> None:
        speclink_token_dense._pending_scores["req0"].append(
            [0.95, 0.95, 0.95, 0.95]
        )
        speclink_token_dense._pending_scores["req1"].append(
            [0.5, 0.99, 0.99, 0.99]
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "3",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": (
                        "balanced_confidence"
                    ),
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {3},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )
            self.assertTrue(speclink_token_dense.draft_scores_required())

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, True, False, False, True]
            + [True, False, False, False, True],
        )

    def test_balanced_confidence_spends_suffix_budget_globally(self) -> None:
        speclink_token_dense._pending_scores["req0"].append(
            [0.99, 0.99, 0.99, 0.99]
        )
        speclink_token_dense._pending_scores["req1"].append(
            [0.20, 0.99, 0.99, 0.99]
        )
        speclink_token_dense._pending_scores["req2"].append(
            [0.10, 0.99, 0.99, 0.99]
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "5",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": (
                        "balanced_confidence"
                    ),
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {5},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1", "req2"],
                num_scheduled_tokens=[5, 5, 5],
                num_draft_tokens=[4, 4, 4],
                num_decode_draft_tokens=[4, 4, 4],
                cu_num_scheduled_tokens=[5, 10, 15],
                total_num_scheduled_tokens=15,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, True, True, False, True]
            + [True, False, False, False, True]
            + [True, False, False, False, True],
        )

    def test_balanced_low_confidence_ranks_uncertain_suffix_first(self) -> None:
        speclink_token_dense._pending_scores["req0"].append(
            [0.95, 0.95, 0.95, 0.95]
        )
        speclink_token_dense._pending_scores["req1"].append(
            [0.5, 0.99, 0.99, 0.99]
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "3",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": (
                        "balanced_low_confidence"
                    ),
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {3},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, False, False, False, True]
            + [True, True, False, False, True],
        )

    def test_request_ranked_budget_protects_two_row_prefixes(self) -> None:
        speclink_token_dense._pending_scores["req0"].append(
            [0.9, 0.9, 0.9, 0.9]
        )
        speclink_token_dense._pending_scores["req1"].append(
            [0.8, 0.9, 0.9, 0.9]
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "2",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "request_highest",
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {2},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, True, False, False, True]
            + [False, False, False, False, True],
        )

    def test_request_contiguous_routes_complete_trailing_requests(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_BUDGET_MODE": "dynamic",
                    "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.5",
                    "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST": "0",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": (
                        "request_contiguous"
                    ),
                },
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )
            self.assertFalse(speclink_token_dense.draft_scores_required())

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.dense_mask.tolist(), [True] * 5 + [False] * 5)
        self.assertEqual(plan.dense_count, 5)
        self.assertEqual(plan.sparse_count, 5)
        self.assertTrue(plan.contiguous_dense_prefix)

    def test_batch_alternating_selects_pure_dual_graph_routes(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_alternating",
                    "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.5",
                },
            ),
        ):
            kwargs = {
                "req_ids": ["req0", "req1"],
                "num_scheduled_tokens": [5, 5],
                "num_draft_tokens": [4, 4],
                "num_decode_draft_tokens": [4, 4],
                "cu_num_scheduled_tokens": [5, 10],
                "total_num_scheduled_tokens": 10,
                "device": torch.device("cpu"),
            }
            dense_plan = speclink_token_dense.build_verify_dense_mask(**kwargs)
            sparse_plan = speclink_token_dense.build_verify_dense_mask(**kwargs)
            self.assertFalse(speclink_token_dense.draft_scores_required())

        self.assertIsNotNone(dense_plan)
        assert dense_plan is not None
        self.assertEqual(dense_plan.dense_count, 10)
        self.assertEqual(dense_plan.sparse_count, 0)
        self.assertTrue(dense_plan.contiguous_dense_prefix)
        self.assertIsNotNone(sparse_plan)
        assert sparse_plan is not None
        self.assertEqual(sparse_plan.dense_count, 0)
        self.assertEqual(sparse_plan.sparse_count, 10)
        with mock.patch.dict(
            os.environ,
            {"SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_alternating"},
        ):
            self.assertEqual(speclink_token_dense.cudagraph_route(sparse_plan), 1)

    def test_batch_adaptive_routes_by_active_request_count(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_adaptive",
                    "SPECLINK_TOKEN_DENSE_ADAPTIVE_DENSE_MAX_REQUESTS": "1",
                },
            ),
        ):
            dense_plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0"],
                num_scheduled_tokens=[5],
                num_draft_tokens=[4],
                num_decode_draft_tokens=[4],
                cu_num_scheduled_tokens=[5],
                total_num_scheduled_tokens=5,
                device=torch.device("cpu"),
            )
            sparse_plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[5, 5],
                num_draft_tokens=[4, 4],
                num_decode_draft_tokens=[4, 4],
                cu_num_scheduled_tokens=[5, 10],
                total_num_scheduled_tokens=10,
                device=torch.device("cpu"),
            )
            self.assertFalse(speclink_token_dense.draft_scores_required())
            self.assertEqual(
                speclink_token_dense.cudagraph_route(sparse_plan), 1
            )

        self.assertIsNotNone(dense_plan)
        self.assertIsNotNone(sparse_plan)
        assert dense_plan is not None and sparse_plan is not None
        self.assertEqual(dense_plan.dense_count, 5)
        self.assertEqual(dense_plan.sparse_count, 0)
        self.assertEqual(sparse_plan.dense_count, 0)
        self.assertEqual(sparse_plan.sparse_count, 10)

    def test_batch_adaptive_prepares_distinct_dense_and_sparse_graphs(self) -> None:
        self.assertIn(
            "batch_adaptive",
            speclink_token_dense.PURE_BATCH_DENSE_SELECTIONS,
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_DENSE_SELECTION",
                "batch_adaptive",
            ),
            mock.patch.object(speclink_token_dense, "_graph_capture_plans", {}),
            mock.patch.object(speclink_token_dense, "_graph_plan_buffers", {}),
            mock.patch.object(speclink_token_dense, "_plan_buffers", {}),
        ):
            dense_plan = speclink_token_dense.prepare_cudagraph_plan(
                18,
                torch.device("cpu"),
                uniform_decode=True,
                route=0,
            )
            sparse_plan = speclink_token_dense.prepare_cudagraph_plan(
                18,
                torch.device("cpu"),
                uniform_decode=True,
                route=1,
            )
            self.assertTrue(
                speclink_token_dense.verify_plan_fits_cudagraph(
                    sparse_plan,
                    actual_rows=18,
                    padded_rows=18,
                )
            )
            self.assertIs(
                speclink_token_dense.pad_verify_plan_for_cudagraph(
                    sparse_plan,
                    actual_rows=18,
                    padded_rows=18,
                    device=torch.device("cpu"),
                ),
                sparse_plan,
            )

        self.assertIsNotNone(dense_plan)
        self.assertIsNotNone(sparse_plan)
        assert dense_plan is not None and sparse_plan is not None
        self.assertEqual((dense_plan.dense_count, dense_plan.sparse_count), (18, 0))
        self.assertEqual((sparse_plan.dense_count, sparse_plan.sparse_count), (0, 18))

    def test_batch_confidence_is_capped_by_sparse_route_budget(self) -> None:
        for _ in range(4):
            speclink_token_dense._pending_scores["req0"].append([0.9] * 4)
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_confidence",
                    "SPECLINK_TOKEN_DENSE_BATCH_CONFIDENCE_THRESHOLD": "0.1",
                    "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.5",
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_dense_override"
                    ),
                },
            ),
        ):
            kwargs = {
                "req_ids": ["req0"],
                "num_scheduled_tokens": [5],
                "num_draft_tokens": [4],
                "num_decode_draft_tokens": [4],
                "cu_num_scheduled_tokens": [5],
                "total_num_scheduled_tokens": 5,
                "device": torch.device("cpu"),
            }
            plans = [
                speclink_token_dense.build_verify_dense_mask(**kwargs)
                for _ in range(4)
            ]

        self.assertIsNotNone(plans[0])
        assert plans[0] is not None
        self.assertEqual(plans[0].dense_count, 5)
        self.assertIsNotNone(plans[1])
        self.assertIsNotNone(plans[2])
        assert plans[2] is not None
        self.assertEqual(plans[2].dense_count, 5)
        self.assertIsNotNone(plans[3])

    def test_batch_alternating_can_hold_routes_in_blocks(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_alternating",
                    "SPECLINK_TOKEN_DENSE_DENSE_RATIO": "0.5",
                    "SPECLINK_TOKEN_DENSE_BATCH_ROUTE_BLOCK_STEPS": "2",
                },
            ),
        ):
            kwargs = {
                "req_ids": ["req0"],
                "num_scheduled_tokens": [5],
                "num_draft_tokens": [4],
                "num_decode_draft_tokens": [4],
                "cu_num_scheduled_tokens": [5],
                "total_num_scheduled_tokens": 5,
                "device": torch.device("cpu"),
            }
            plans = [
                speclink_token_dense.build_verify_dense_mask(**kwargs)
                for _ in range(4)
            ]

        self.assertIsNotNone(plans[0])
        self.assertIsNotNone(plans[1])
        assert plans[0] is not None
        assert plans[1] is not None
        self.assertEqual(plans[0].dense_count, 5)
        self.assertEqual(plans[1].dense_count, 5)
        self.assertIsNotNone(plans[2])
        self.assertIsNotNone(plans[3])

    def test_confidence_ranking_uses_probability_of_reaching_row(self) -> None:
        speclink_token_dense._pending_scores["req0"].append([0.9, 0.1, 0.9])
        speclink_token_dense._pending_scores["req1"].append([0.8, 0.99, 0.9])
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "1",
                    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "highest",
                },
            ),
            mock.patch.object(
                speclink_token_dense,
                "VALID_DENSE_TOKEN_BUDGETS",
                speclink_token_dense.VALID_DENSE_TOKEN_BUDGETS | {1},
            ),
        ):
            plan = speclink_token_dense.build_verify_dense_mask(
                req_ids=["req0", "req1"],
                num_scheduled_tokens=[4, 4],
                num_draft_tokens=[3, 3],
                num_decode_draft_tokens=[3, 3],
                cu_num_scheduled_tokens=[4, 8],
                total_num_scheduled_tokens=8,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [True, False, False, True, False, False, False, True],
        )

    def test_token_dense_accuracy_forwards_fp16_to_vllm(self) -> None:
        args = run_token_dense_accuracy.build_parser().parse_args([])

        command = run_structured_24_spec_quality.build_vllm_command(
            args,
            base_model="base",
            speculator_model="draft",
            port=8000,
        )

        dtype_index = command.index("--dtype")
        self.assertEqual(command[dtype_index + 1], "float16")

    def test_dynamic_token_routing_supports_graph_safe_decode_capture(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])

        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d16",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )

        config_index = command.index("--compilation-config")
        config = json.loads(command[config_index + 1])
        self.assertEqual(config, {"mode": "NONE", "cudagraph_mode": "NONE"})
        self.assertNotIn("--enforce-eager", command)

        args.token_dense_cudagraph_mode = "full_decode_only"
        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d16",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )
        config_index = command.index("--compilation-config")
        config = json.loads(command[config_index + 1])
        self.assertEqual(
            config,
            {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
        )
        method = token_dense_methods.parse_method_config("token_dense_d16")
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_GRAPH_ROUTING"], "1")
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_NUM_SPEC_TOKENS"], "8")
        self.assertEqual(env["SPECLINK_DECODE_ONLY_ISOLATE_BATCHES"], "0")
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_PROJECTION_POLICY"],
            "all",
        )

        args.token_dense_release_dense_weights = True
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT"],
            "1",
        )

        args.token_dense_cudagraph_mode = "full"
        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d16",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )
        config_index = command.index("--compilation-config")
        self.assertEqual(
            json.loads(command[config_index + 1]),
            {"mode": "NONE", "cudagraph_mode": "FULL_AND_PIECEWISE"},
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_GRAPH_ROUTING"], "1")

        args.token_dense_isolate_decode_batches = True
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )
        self.assertEqual(env["SPECLINK_DECODE_ONLY_ISOLATE_BATCHES"], "1")

        self.assertNotIn("--no-enable-flashinfer-autotune", command)
        args.token_dense_flashinfer_autotune = False
        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d16",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )
        self.assertIn("--no-enable-flashinfer-autotune", command)

    def test_sparse_only_decode_supports_full_decode_graph(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        args.production_fast = True
        args.token_dense_linear_strategy = "sparse_only_decode"
        args.token_dense_cudagraph_mode = "full_decode_only"

        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d0",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )

        config_index = command.index("--compilation-config")
        config = json.loads(command[config_index + 1])
        self.assertEqual(
            config,
            {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
        )

        args.token_dense_compilation_mode = "vllm_compile"
        command = run_lm_eval_accuracy.build_vllm_command(
            args,
            mode="token_dense_d0",
            model_path="base",
            speculator_path="draft",
            port=8000,
        )
        config_index = command.index("--compilation-config")
        config = json.loads(command[config_index + 1])
        self.assertEqual(
            config,
            {
                "mode": "VLLM_COMPILE",
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
        )

    def test_lm_eval_token_dense_layer_policy_reaches_method_env(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        args.token_dense_layer_policy = "keep_first_last"
        args.token_dense_keep_n = 2
        method = run_lm_eval_accuracy.mode_method("token_dense_d16")
        method = replace(
            method,
            policy=args.token_dense_layer_policy,
            keep_n=args.token_dense_keep_n,
        )
        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )

        self.assertEqual(env["SPECLINK_STRUCTURED_24_POLICY"], "keep_first_last")
        self.assertEqual(env["SPECLINK_STRUCTURED_24_KEEP_N"], "2")

    def test_sparse_value_scale_reaches_prepack_env(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        args.token_dense_sparse_value_scale = 1.125
        args.token_dense_row_scale_mode = "variance"
        args.token_dense_variance_scale_projection_policy = "gate_up"
        args.token_dense_row_scale_max = 1.5
        args.token_dense_sparse_output_mode = "view_mlp_o"
        method = run_lm_eval_accuracy.mode_method("token_dense_d0")

        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )

        self.assertEqual(env["SPECLINK_SPARSE24_VALUE_SCALE"], "1.125")
        self.assertEqual(env["SPECLINK_SPARSE24_ROW_SCALE_MODE"], "variance")
        self.assertEqual(
            env["SPECLINK_SPARSE24_VARIANCE_SCALE_PROJECTION_POLICY"],
            "gate_up",
        )
        self.assertEqual(env["SPECLINK_SPARSE24_ROW_SCALE_MAX"], "1.5")
        self.assertEqual(env["SPECLINK_SPARSE24_SKIP_TRANSPOSE"], "layerwise")
        self.assertEqual(env["SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT"], "1")

    def test_fused_mlp_output_mode_reaches_runtime_env(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        args.token_dense_sparse_output_mode = "fused_mlp"
        args.token_dense_sparse_accumulator = "fp16_gate_down"
        args.token_dense_gate_up_dense_layers = "4,10,24-25"
        args.token_dense_gate_up_value_scale = 1.05
        args.token_dense_group_reconstruction = True
        method = run_lm_eval_accuracy.mode_method("token_dense_d0")

        env = token_dense_methods.method_env(
            args,
            model_label="qwen3_8b",
            method=method,
            stats_path=Path("/tmp/token_dense_stats.json"),
        )

        self.assertEqual(env["SPECLINK_SPARSE24_SKIP_TRANSPOSE"], "gate_up")
        self.assertEqual(env["SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT"], "1")
        self.assertEqual(env["SPECLINK_SPARSE24_TRANSPOSED_MLP_FUSION"], "1")
        self.assertEqual(
            env["SPECLINK_SPARSE24_ACCUMULATOR"], "fp16_gate_down"
        )
        self.assertEqual(
            env["SPECLINK_TOKEN_DENSE_GATE_UP_DENSE_LAYERS"],
            "4,10,24-25",
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_GATE_UP_VALUE_SCALE"], "1.05"
        )
        self.assertEqual(
            env["SPECLINK_SPARSE24_GROUP_RECONSTRUCTION"], "1"
        )
        self.assertTrue(
            env["SPECLINK_SPARSE24_GROUP_COVARIANCE_CACHE"].endswith(
                "/qwen3_8b/gate_group_covariances.pt"
            )
        )
        self.assertEqual(env["VLLM_DISABLE_COMPILE_CACHE"], "1")

    def test_gate_up_dense_layer_parser_supports_ranges(self) -> None:
        self.assertEqual(
            speclink_structured_24._parse_layer_indices("4,10,24-25"),
            {4, 10, 24, 25},
        )
        with self.assertRaises(ValueError):
            speclink_structured_24._parse_layer_indices("-1")

    def test_qwen_down_static_sparse_uses_dense_below_break_even(self) -> None:
        self.assertTrue(
            speclink_linear._prefer_dense_static_projection(
                40,
                in_features=12288,
                out_features=4096,
            )
        )
        self.assertFalse(
            speclink_linear._prefer_dense_static_projection(
                48,
                in_features=12288,
                out_features=4096,
            )
        )
        self.assertFalse(
            speclink_linear._prefer_dense_static_projection(
                40,
                in_features=4096,
                out_features=4096,
            )
        )
        self.assertTrue(
            speclink_linear._prefer_dense_mixed_prefill_projection(
                in_features=12288,
                out_features=4096,
            )
        )
        self.assertFalse(
            speclink_linear._prefer_dense_mixed_prefill_projection(
                in_features=4096,
                out_features=6144,
            )
        )

    def test_covwanda_mask_uses_grouped_quadratic_error(self) -> None:
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        covariance = torch.eye(4).unsqueeze(0)

        group_bytes = prepare_covwanda_gate_masks.covariance_mask(
            weight,
            covariance,
            row_chunk=1,
        )

        self.assertEqual(group_bytes.tolist(), [[0xC]])

    def test_grouped_reconstruction_folds_correlated_removed_inputs(self) -> None:
        weight = torch.ones((1, 4), dtype=torch.float16)
        group_bytes = torch.tensor([[0x3]], dtype=torch.uint8)
        covariance = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.9, 0.0],
                    [0.0, 1.0, 0.0, 0.9],
                    [0.9, 0.0, 1.0, 0.0],
                    [0.0, 0.9, 0.0, 1.0],
                ]
            ]
        )

        reconstructed, stats = (
            speclink_structured_24._reconstruct_grouped_24_weight(
                weight,
                group_bytes,
                covariance,
            )
        )

        self.assertTrue(
            torch.allclose(
                reconstructed.float(),
                torch.tensor([[1.9, 1.9, 0.0, 0.0]]),
                atol=2e-3,
                rtol=2e-3,
            )
        )
        self.assertLess(stats["group_reconstruction_error_ratio"], 1.0)

    def test_variance_row_scale_preserves_weighted_second_moment(self) -> None:
        weight = torch.ones((2, 4), dtype=torch.float16)
        group_bytes = torch.full((2, 1), 0b0011, dtype=torch.uint8)
        activation_scale = torch.ones(4, dtype=torch.float32)

        scale = speclink_structured_24._variance_preserving_row_scale(
            weight,
            group_bytes,
            activation_scale,
            max_scale=2.0,
        )

        expected = torch.full((2,), 2.0**0.5, dtype=torch.float16)
        self.assertTrue(torch.allclose(scale, expected, atol=1e-3, rtol=1e-3))

    def test_sparse_value_scale_is_applied_during_prepack(self) -> None:
        weight = torch.ones((32, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((32, 32), dtype=torch.float16),
            meta=torch.zeros((32, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock(
            return_value=(
                torch.ones((32, 32), dtype=torch.float16),
                torch.zeros(128, dtype=torch.uint16),
            )
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_SPARSE24_VALUE_SCALE": "1.125",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "test.linear",
                weight,
                {},
            )

        scale = pack.call_args.args[2]
        self.assertTrue(torch.equal(scale, torch.full((32,), 1.125)))

    def test_inline_swiglu_uses_interleaved_gate_up_prepack(self) -> None:
        weight = torch.ones((256, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_selective_mixed_rows=True,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((256, 32), dtype=torch.float16),
            meta=torch.zeros((256, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        interleaved_values = torch.full((256, 32), 2.0, dtype=torch.float16)
        interleaved_meta = torch.full((256, 4), 3, dtype=torch.int16)
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock()
        prepare_swiglu = mock.Mock(
            return_value=(interleaved_values, interleaved_meta)
        )
        stats: dict[str, object] = {}

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_dense_override"
                    ),
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP": "1",
                    "SPECLINK_TOKEN_DENSE_INLINE_SWIGLU_MLP": "1",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
            mock.patch(
                "vllm.speclink_kernel.prepare_cutlass_sparse24_gate_up_swiglu",
                prepare_swiglu,
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.0.mlp.gate_up_proj",
                weight,
                stats,
            )

        prepare_swiglu.assert_called_once_with(
            packed.values,
            packed.meta,
            layout="n_major",
            K=64,
        )
        prepare.assert_not_called()
        self.assertTrue(module._speclink_sparse24_gate_up_interleaved)
        self.assertIs(
            module._speclink_sparse24_full_a_values,
            interleaved_values,
        )
        self.assertIs(module._speclink_sparse24_full_a_meta_e, interleaved_meta)
        self.assertEqual(
            stats["speclink_kernel_inline_swiglu_mlp_module_names"],
            ["model.layers.0.mlp.gate_up_proj"],
        )

    def test_sparse_gate_dense_down_uses_interleaved_fixed_gate_prepack(
        self,
    ) -> None:
        weight = torch.ones((256, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_selective_mixed_rows=False,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((256, 32), dtype=torch.float16),
            meta=torch.zeros((256, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        interleaved_values = torch.full((256, 32), 2.0, dtype=torch.float16)
        interleaved_meta = torch.full((256, 4), 3, dtype=torch.int16)
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock()
        prepare_swiglu = mock.Mock(
            return_value=(interleaved_values, interleaved_meta)
        )
        stats: dict[str, object] = {}

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_dense_override"
                    ),
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_TOKEN_DENSE_SPARSE_GATE_DENSE_DOWN": "1",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
            mock.patch(
                "vllm.speclink_kernel.prepare_cutlass_sparse24_gate_up_swiglu",
                prepare_swiglu,
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.0.mlp.gate_up_proj",
                weight,
                stats,
            )

        prepare_swiglu.assert_called_once_with(
            packed.values,
            packed.meta,
            layout="n_major",
            K=64,
        )
        prepare.assert_not_called()
        self.assertTrue(module._speclink_sparse24_gate_up_interleaved)
        self.assertTrue(module._speclink_sparse24_sparse_gate_dense_down)
        self.assertIs(
            module._speclink_sparse24_full_a_values,
            interleaved_values,
        )
        self.assertEqual(
            stats["speclink_kernel_sparse_gate_dense_down_module_names"],
            ["model.layers.0.mlp.gate_up_proj"],
        )

    def test_routed_swiglu_uses_interleaved_full_and_regular_residual(self) -> None:
        weight = torch.ones((256, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_selective_mixed_rows=True,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((256, 32), dtype=torch.float16),
            meta=torch.zeros((256, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        interleaved_values = torch.full((256, 32), 2.0, dtype=torch.float16)
        interleaved_meta = torch.full((256, 4), 3, dtype=torch.int16)
        residual_values = torch.full((256, 32), 4.0, dtype=torch.float16)
        residual_meta = torch.full((256, 4), 5, dtype=torch.int16)
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock(return_value=(residual_values, residual_meta))
        prepare_swiglu = mock.Mock(
            return_value=(interleaved_values, interleaved_meta)
        )
        stats: dict[str, object] = {}

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP": "1",
                    "SPECLINK_TOKEN_DENSE_ROUTED_SWIGLU_MLP": "1",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
            mock.patch(
                "vllm.speclink_kernel.prepare_cutlass_sparse24_gate_up_swiglu",
                prepare_swiglu,
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.0.mlp.gate_up_proj",
                weight,
                stats,
            )

        self.assertEqual(pack.call_count, 2)
        prepare_swiglu.assert_called_once()
        prepare.assert_called_once()
        self.assertTrue(module._speclink_sparse24_gate_up_interleaved)
        self.assertTrue(module._speclink_sparse24_routed_swiglu)
        self.assertIs(module._speclink_sparse24_full_a_values, interleaved_values)
        self.assertIs(module._speclink_sparse24_residual_a_values, residual_values)
        self.assertEqual(
            stats["speclink_kernel_routed_swiglu_mlp_module_names"],
            ["model.layers.0.mlp.gate_up_proj"],
        )

    def test_up_sparse_hybrid_prepacks_only_up_half(self) -> None:
        weight = torch.arange(64 * 64, dtype=torch.float16).view(64, 64)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((32, 32), dtype=torch.float16),
            meta=torch.zeros((32, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock(
            return_value=(
                torch.ones((32, 32), dtype=torch.float16),
                torch.zeros(128, dtype=torch.uint16),
            )
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
                    "SPECLINK_SPARSE24_GATE_UP_HYBRID": "up_sparse",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.0.mlp.gate_up_proj",
                weight,
                {},
            )

        packed_weight, packed_mask, _packed_scale = pack.call_args.args
        self.assertTrue(torch.equal(packed_weight, weight[32:]))
        self.assertEqual(tuple(packed_mask.shape), (32, 16))
        self.assertEqual(module._speclink_gate_up_hybrid, "up_sparse")
        self.assertFalse(module._speclink_gate_up_hybrid_sparse_first)

    def test_sparse_value_scale_rejects_exact_residual_strategy(self) -> None:
        weight = torch.ones((32, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            _speclink_selective_dense_enabled=False,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                    "SPECLINK_SPARSE24_VALUE_SCALE": "1.05",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "row_scale_residual_is_not_exact_2to4",
            ):
                speclink_structured_24._attach_speclink_kernel_prepack(
                    module,
                    "test.linear",
                    weight,
                    {},
                )

    def test_static_projection_release_prepacks_exact_fallback(self) -> None:
        weight = torch.ones((32, 64), dtype=torch.float16)
        keep, _ = speclink_structured_24._compute_keep_mask_24(weight, None)
        module = SimpleNamespace(
            weight=weight,
            _speclink_selective_dense_enabled=False,
            _speclink_selective_mixed_rows=False,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        packed = SimpleNamespace(
            values=torch.ones((32, 32), dtype=torch.float16),
            meta=torch.zeros((32, 16), dtype=torch.uint8),
            layout="n_major",
            K=64,
        )
        full_values = torch.full((32, 32), 2.0, dtype=torch.float16)
        full_meta = torch.full((32, 4), 3, dtype=torch.int16)
        residual_values = torch.full((32, 32), 4.0, dtype=torch.float16)
        residual_meta = torch.full((32, 4), 5, dtype=torch.int16)
        pack = mock.Mock(return_value=packed)
        prepare = mock.Mock(
            side_effect=[
                (full_values, full_meta),
                (residual_values, residual_meta),
            ]
        )
        stats: dict[str, object] = {}

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                    "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT": "1",
                },
                clear=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_cutlass_supported_weight",
                return_value=True,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
                return_value=(pack, prepare),
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.0.self_attn.q_proj",
                weight,
                stats,
            )

        self.assertEqual(pack.call_count, 2)
        self.assertEqual(prepare.call_count, 2)
        self.assertIs(module._speclink_sparse24_full_a_values, full_values)
        self.assertIs(
            module._speclink_sparse24_residual_a_values,
            residual_values,
        )
        self.assertEqual(weight.numel(), 0)
        self.assertTrue(module._speclink_sparse24_dense_weight_released)
        self.assertEqual(
            stats["speclink_kernel_residual_prepack_module_names"],
            ["model.layers.0.self_attn.q_proj"],
        )

    def test_lm_eval_aggregator_receives_external_baseline(self) -> None:
        with mock.patch.object(run_lm_eval_accuracy.subprocess, "run") as run:
            run_lm_eval_accuracy.aggregate_outputs(
                Path("/tmp/candidate"),
                Path("/tmp/baseline"),
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--baseline-dir") + 1],
            "/tmp/baseline",
        )

    def test_lm_eval_http_client_does_not_claim_gpu(self) -> None:
        args = run_lm_eval_accuracy.build_parser().parse_args([])
        run_dir = Path("/tmp/speclink_lm_eval_client_env")
        run_dir.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(
            run_lm_eval_accuracy.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            rc = run_lm_eval_accuracy.run_lm_eval(
                args,
                task="gsm8k_cot",
                mode="token_dense_d0",
                model_path="base",
                tokenizer_path="base",
                port=8000,
                run_dir=run_dir,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_sparse_only_graph_plan_routes_every_row_sparse(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_LINEAR_STRATEGY",
                "sparse_only_decode",
            ),
        ):
            self.assertEqual(speclink_token_dense._graph_sparse_count(288), 288)

            eager_plan = speclink_token_dense._make_plan(
                torch.empty(0, dtype=torch.bool),
                dense_count=0,
                sparse_count=7,
                total_rows=7,
                all_sparse=True,
            )
            graph_plan = speclink_token_dense.pad_verify_plan_for_cudagraph(
                eager_plan,
                actual_rows=7,
                padded_rows=8,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(graph_plan)
        assert graph_plan is not None
        self.assertEqual(graph_plan.dense_count, 0)
        self.assertEqual(graph_plan.sparse_count, 8)
        self.assertEqual(graph_plan.dense_mask.tolist(), [False] * 8)

    def test_graph_sparse_count_includes_sparse_bonus_rows(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_DENSE_SELECTION",
                "balanced_confidence",
            ),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_LINEAR_STRATEGY",
                "full_sparse_residual",
            ),
            mock.patch.object(speclink_token_dense, "_STATIC_SPARSE_BONUS", True),
            mock.patch.object(
                speclink_token_dense,
                "effective_dense_token_budget",
                return_value=32,
            ),
        ):
            self.assertEqual(speclink_token_dense._graph_sparse_count(288), 256)

        with mock.patch.object(
            speclink_token_dense,
            "_STATIC_SPARSE_BONUS",
            False,
        ):
            with mock.patch.object(
                speclink_token_dense,
                "effective_dense_token_budget",
                return_value=32,
            ):
                self.assertEqual(speclink_token_dense._graph_sparse_count(288), 224)

    def test_graph_plan_uses_fixed_sparse_count_and_padding_rows(self) -> None:
        dense_mask = torch.ones(44, dtype=torch.bool)
        dense_mask[8:44] = False
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=8,
            sparse_count=36,
            total_rows=44,
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0"},
            ),
        ):
            graph_plan = speclink_token_dense.pad_verify_plan_for_cudagraph(
                plan,
                actual_rows=44,
                padded_rows=48,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(graph_plan)
        assert graph_plan is not None
        self.assertEqual(graph_plan.total_rows, 48)
        self.assertEqual(graph_plan.dense_count, 8)
        self.assertEqual(graph_plan.sparse_count, 40)
        self.assertEqual(
            graph_plan.sparse_rows[: graph_plan.sparse_count].tolist(),
            list(range(8, 48)),
        )

    def test_plan_reuses_compact_dense_slot_mapping(self) -> None:
        dense_mask = torch.tensor([False, True, False, True, True, False])
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=3,
            sparse_count=3,
            total_rows=6,
        )

        self.assertIsNotNone(plan.dense_slots)
        assert plan.dense_slots is not None
        self.assertEqual(plan.dense_slots.tolist(), [-1, 0, -1, 1, 2, -1])
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                slots = speclink_token_dense.current_verify_dense_slots(
                    6, torch.device("cpu")
                )
            finally:
                speclink_token_dense.end_verify_context(token)

        self.assertIsNotNone(slots)
        assert slots is not None
        self.assertEqual(slots.data_ptr(), plan.dense_slots.data_ptr())

    def test_graph_plan_rejects_unpadded_dense_fallback(self) -> None:
        dense_mask = torch.ones(288, dtype=torch.bool)
        dense_mask[32:] = False
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=32,
            sparse_count=256,
            total_rows=288,
        )
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0"},
            ),
        ):
            self.assertTrue(
                speclink_token_dense.verify_plan_fits_cudagraph(
                    plan,
                    actual_rows=288,
                    padded_rows=288,
                )
            )
            self.assertFalse(
                speclink_token_dense.verify_plan_fits_cudagraph(
                    None,
                    actual_rows=288,
                    padded_rows=288,
                )
            )

    def test_graph_plan_rejects_mixed_prefill_decode_batch(self) -> None:
        plan = speclink_token_dense._make_plan(
            torch.tensor([False, False, True, True]),
            dense_count=2,
            sparse_count=2,
            total_rows=4,
            has_prefill_rows=True,
        )
        with mock.patch.object(
            speclink_token_dense,
            "_STATIC_GRAPH_ROUTING",
            True,
        ):
            self.assertFalse(
                speclink_token_dense.verify_plan_fits_cudagraph(
                    plan,
                    actual_rows=4,
                    padded_rows=8,
                )
            )

    def test_graph_capture_and_runtime_plan_share_exact_buffers(self) -> None:
        with (
            mock.patch.object(
                speclink_token_dense,
                "_full_cudagraph_context_active",
                return_value=True,
            ),
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0"},
            ),
        ):
            capture_summary = (
                speclink_token_dense.current_verify_dense_row_summary(
                    48,
                    torch.device("cpu"),
                )
            )
            self.assertIsNotNone(capture_summary)
            assert capture_summary is not None
            capture_dense_rows = capture_summary[2]

            dense_mask = torch.ones(44, dtype=torch.bool)
            dense_mask[8:44] = False
            runtime_plan = speclink_token_dense._make_plan(
                dense_mask,
                dense_count=8,
                sparse_count=36,
                total_rows=44,
            )
            padded_plan = speclink_token_dense.pad_verify_plan_for_cudagraph(
                runtime_plan,
                actual_rows=44,
                padded_rows=48,
                device=torch.device("cpu"),
            )

        self.assertIsNotNone(padded_plan)
        assert padded_plan is not None
        self.assertEqual(
            capture_dense_rows.data_ptr(),
            padded_plan.dense_rows.data_ptr(),
        )

    def test_verify_summary_marks_backend_alignment_rows_sparse(self) -> None:
        dense_mask = torch.zeros(434, dtype=torch.bool)
        dense_mask[:8] = True
        with mock.patch.object(
            speclink_token_dense,
            "_STATIC_ENABLED",
            True,
        ):
            token = speclink_token_dense.begin_verify_context(dense_mask)
            try:
                summary = speclink_token_dense.current_verify_dense_row_summary(
                    448,
                    torch.device("cpu"),
                )
            finally:
                speclink_token_dense.end_verify_context(token)

        self.assertIsNotNone(summary)
        assert summary is not None
        row_is_dense, dense_count, dense_rows, sparse_rows = summary
        self.assertEqual(tuple(row_is_dense.shape), (448,))
        self.assertEqual(dense_count, 8)
        self.assertEqual(dense_rows.tolist(), list(range(8)))
        self.assertEqual(sparse_rows[-14:].tolist(), list(range(434, 448)))

    def test_verify_summary_rejects_non_alignment_size_mismatch(self) -> None:
        dense_mask = torch.zeros(432, dtype=torch.bool)
        with mock.patch.object(
            speclink_token_dense,
            "_STATIC_ENABLED",
            True,
        ):
            token = speclink_token_dense.begin_verify_context(dense_mask)
            try:
                with self.assertRaisesRegex(RuntimeError, "mask has 432 rows"):
                    speclink_token_dense.current_verify_dense_row_summary(
                        448,
                        torch.device("cpu"),
                    )
            finally:
                speclink_token_dense.end_verify_context(token)

    def test_triton_score_backend_precomputes_prefix_log_confidence(self) -> None:
        logits_by_position = [
            torch.tensor([[2.0, 1.0], [1.0, 3.0]]),
            torch.tensor([[4.0, 0.0], [2.0, 1.0]]),
        ]
        draft_token_ids = torch.tensor([[0, 0], [1, 0]])
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_SCORE_BACKEND": "triton_selected"},
            ),
            mock.patch.object(
                speclink_token_dense,
                "_compute_selected_logprobs",
                side_effect=lambda logits, selected: torch.log_softmax(
                    logits.float(), dim=-1
                )
                .gather(1, selected.view(-1, 1))
                .squeeze(1),
            ),
        ):
            context_token = speclink_token_dense.begin_propose_context(
                req_ids=["req0", "req1"],
                prompt_lens=[1, 1],
                generated_lens=[1, 1],
                active_requests=2,
                batch_size=2,
                num_spec_tokens=2,
            )
            try:
                speclink_token_dense.record_draft_scores(
                    draft_token_ids=draft_token_ids,
                    logits_by_position=logits_by_position,
                )
            finally:
                speclink_token_dense.end_propose_context(context_token)

        req0 = speclink_token_dense._pending_scores["req0"][0]
        req1 = speclink_token_dense._pending_scores["req1"][0]
        self.assertIsInstance(req0, speclink_token_dense._CumulativeLogScores)
        self.assertIsInstance(req1, speclink_token_dense._CumulativeLogScores)
        selected_logprobs = torch.stack(
            [
                torch.log_softmax(logits_by_position[0], dim=-1)[
                    torch.arange(2), draft_token_ids[:, 0]
                ],
                torch.log_softmax(logits_by_position[1], dim=-1)[
                    torch.arange(2), draft_token_ids[:, 1]
                ],
            ],
            dim=1,
        ).cumsum(dim=1)
        self.assertTrue(torch.allclose(req0.values, selected_logprobs[0]))
        self.assertTrue(torch.allclose(req1.values, selected_logprobs[1]))

    def test_fused_score_backend_accepts_precomputed_logprobs(self) -> None:
        selected_logprobs = [
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-0.3, -0.4]),
        ]
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_SCORE_BACKEND": "triton_fused"},
            ),
        ):
            context_token = speclink_token_dense.begin_propose_context(
                req_ids=["req0", "req1"],
                prompt_lens=[1, 1],
                generated_lens=[1, 1],
                active_requests=2,
                batch_size=2,
                num_spec_tokens=2,
            )
            try:
                speclink_token_dense.record_draft_scores(
                    draft_token_ids=torch.tensor([[0, 1], [1, 0]]),
                    logits_by_position=[],
                    selected_logprobs_by_position=selected_logprobs,
                )
            finally:
                speclink_token_dense.end_propose_context(context_token)

        req0 = speclink_token_dense._pending_scores["req0"][0]
        req1 = speclink_token_dense._pending_scores["req1"][0]
        self.assertTrue(
            torch.allclose(req0.values, torch.tensor([-0.1, -0.4]))
        )
        self.assertTrue(
            torch.allclose(req1.values, torch.tensor([-0.2, -0.6]))
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_greedy_logprobs_match_torch(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn((4, 4097), device="cuda", dtype=torch.float16)
        expected_ids = logits.argmax(dim=-1)
        expected_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            1,
            expected_ids.view(-1, 1),
        ).squeeze(1)

        actual_ids, actual_logprobs = (
            speclink_token_dense.compute_greedy_token_ids_and_logprobs(logits)
        )

        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(
            torch.allclose(actual_logprobs, expected_logprobs, atol=2e-6, rtol=0)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_greedy_logprobs_ignore_all_negative_inf_blocks(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn((2, 4097), device="cuda", dtype=torch.float16)
        logits[:, 1024:2048] = float("-inf")
        expected_ids = logits.argmax(dim=-1)
        expected_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            1,
            expected_ids.view(-1, 1),
        ).squeeze(1)

        actual_ids, actual_logprobs = (
            speclink_token_dense.compute_greedy_token_ids_and_logprobs(logits)
        )

        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.isfinite(actual_logprobs).all())
        self.assertTrue(
            torch.allclose(actual_logprobs, expected_logprobs, atol=2e-6, rtol=0)
        )

    def test_decode_sparse_mode_routes_all_decode_rows_sparse(self) -> None:
        self.assertEqual(
            offline_24_pruning.decode_sparse_lm_eval_modes(
                "dense_ar,eagle3_dense"
            ),
            "token_dense_d0",
        )
        method = token_dense_methods.parse_method_config("token_dense_d0")
        self.assertEqual(method.token_dense_budget, 0)
        with mock.patch.dict(
            os.environ,
            {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0"},
        ):
            self.assertEqual(speclink_token_dense.dense_token_budget(), 0)
        self.assertEqual(
            offline_24_pruning.lm_eval_modes_for_method(
                "dense_ar,eagle3_dense",
                decode_sparse_only=True,
                method="original",
            ),
            "dense_ar,eagle3_dense",
        )

    def test_inherited_offline_mask_is_not_replaced_by_wanda(self) -> None:
        inherited_cache = "/tmp/proxsparse.pt"
        args = SimpleNamespace(
            calibration_cache_root=Path("/tmp/calibration"),
            token_dense_mask_method="inherit",
            token_dense_mask_root=Path("/tmp/default_masks"),
            token_dense_linear_strategy="sparse_only_decode",
            token_dense_mlp_strategy="auto",
            production_fast=False,
        )
        method = token_dense_methods.parse_method_config("token_dense_d0")
        with mock.patch.dict(
            os.environ,
            {
                "SPECLINK_STRUCTURED_24_MASK_CACHE": inherited_cache,
                "SPECLINK_STRUCTURED_24_CACHE_STRICT": "1",
            },
        ):
            env = token_dense_methods.method_env(
                args,
                model_label="llama3_1_8b",
                method=method,
                stats_path=Path("/tmp/token_dense_stats.json"),
            )

        self.assertEqual(env["SPECLINK_STRUCTURED_24_MASK_CACHE"], inherited_cache)
        self.assertEqual(env["SPECLINK_TOKEN_DENSE_DENSE_TOKENS"], "0")

    def test_offline_decode_command_selects_cache_and_sparse_only_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_path = root / "masks" / "qwen3_8b" / "proxsparse.pt"
            mask_path.parent.mkdir(parents=True)
            mask_path.touch()
            args = offline_24_pruning.build_parser().parse_args(
                [
                    "run-lm-eval",
                    "--models",
                    "qwen3_8b",
                    "--methods",
                    "proxsparse,original",
                    "--model-id",
                    "qwen3_8b=/tmp/base",
                    "--mask-root",
                    str(root / "masks"),
                    "--output-root",
                    str(root / "output"),
                    "--decode-sparse-only",
                ]
            )
            completed = SimpleNamespace(returncode=0)
            with (
                mock.patch.object(
                    offline_24_pruning.subprocess,
                    "run",
                    return_value=completed,
                ) as run_mock,
                mock.patch.object(
                    offline_24_pruning,
                    "write_combined_summary",
                    return_value=[],
                ),
                mock.patch.object(offline_24_pruning, "write_accuracy_plots"),
                mock.patch("builtins.print"),
            ):
                offline_24_pruning.run_lm_eval(args)

        calls = run_mock.call_args_list
        self.assertEqual(len(calls), 2)
        commands = [call.args[0] for call in calls]
        sparse_command = next(
            command
            for command in commands
            if command[command.index("--mode") + 1] == "token_dense_d0"
        )
        original_command = next(
            command
            for command in commands
            if command[command.index("--mode") + 1] == "dense_ar,eagle3_dense"
        )
        self.assertEqual(
            sparse_command[sparse_command.index("--token-dense-mask-method") + 1],
            "inherit",
        )
        self.assertEqual(
            sparse_command[
                sparse_command.index("--token-dense-linear-strategy") + 1
            ],
            "sparse_only_decode",
        )
        self.assertNotIn("--token-dense-mask-method", original_command)
        sparse_call = next(
            call
            for call in calls
            if call.args[0][call.args[0].index("--mode") + 1] == "token_dense_d0"
        )
        self.assertEqual(
            sparse_call.kwargs["env"]["SPECLINK_STRUCTURED_24_MASK_CACHE"],
            str(mask_path.resolve()),
        )

    def test_gate_only_sparse_decode_does_not_prepack_down_proj(self) -> None:
        down_proj = torch.nn.Linear(4, 4, bias=False)
        module_name = "model.layers.0.mlp.down_proj"
        env = {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": "test_model",
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": "/tmp/calibration",
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
            "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "gate_only",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                speclink_structured_24,
                "_iter_target_modules",
                return_value=[(module_name, down_proj, down_proj.weight)],
            ),
            mock.patch.object(
                speclink_structured_24,
                "_load_activation_scales",
                return_value={},
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
            ),
        ):
            stats = speclink_structured_24.apply_structured_24_from_env(object())

        assert stats is not None
        self.assertEqual(stats["masked_module_names"], [])
        self.assertEqual(stats["dense_keep_module_names"], [module_name])
        self.assertEqual(stats["speclink_kernel_prepack_module_count"], 0)
        self.assertTrue(down_proj._speclink_selective_dense_bypass)
        self.assertEqual(
            stats["per_module"][0]["mask_method"],
            "token_dense_mlp_gate_only_down_dense",
        )

    def test_token_dense_layer_policy_keeps_selected_layers_dense(self) -> None:
        first = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        second = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        first_name = "model.layers.0.self_attn.q_proj"
        second_name = "model.layers.1.self_attn.q_proj"
        env = {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": "test_model",
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": "/tmp/calibration",
            "SPECLINK_STRUCTURED_24_POLICY": "keep_first",
            "SPECLINK_STRUCTURED_24_KEEP_N": "1",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
            "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
        }
        modules = [
            (first_name, first, first.weight),
            (second_name, second, second.weight),
        ]
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                speclink_structured_24,
                "_iter_target_modules",
                return_value=modules,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_load_activation_scales",
                return_value={},
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
            ),
            mock.patch.object(
                speclink_structured_24,
                "_attach_speclink_kernel_prepack",
            ),
        ):
            stats = speclink_structured_24.apply_structured_24_from_env(object())

        assert stats is not None
        self.assertEqual(stats["dense_keep_module_names"], [first_name])
        self.assertEqual(stats["masked_module_names"], [second_name])
        self.assertTrue(first._speclink_selective_dense_bypass)
        self.assertIsInstance(second._speclink_24_mask_bytes, torch.Tensor)

    def test_layerwise_mixed_policy_marks_other_modules_pure_sparse(self) -> None:
        first = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        second = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        modules = [
            ("model.layers.0.self_attn.q_proj", first, first.weight),
            ("model.layers.1.self_attn.q_proj", second, second.weight),
        ]
        env = {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": "test_model",
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": "/tmp/calibration",
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                "full_sparse_dense_override"
            ),
            "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
            "SPECLINK_TOKEN_DENSE_MIXED_LAYERS": "1",
            "SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY": "qkv",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                speclink_structured_24,
                "_iter_target_modules",
                return_value=modules,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_load_activation_scales",
                return_value={},
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
            ),
            mock.patch.object(
                speclink_structured_24,
                "_attach_speclink_kernel_prepack",
            ),
        ):
            stats = speclink_structured_24.apply_structured_24_from_env(object())

        assert stats is not None
        self.assertFalse(first._speclink_selective_mixed_rows)
        self.assertTrue(second._speclink_selective_mixed_rows)
        self.assertEqual(
            stats["speclink_kernel_sparse_only_module_names"],
            ["model.layers.0.self_attn.q_proj"],
        )
        self.assertEqual(
            stats["speclink_kernel_mixed_module_names"],
            ["model.layers.1.self_attn.q_proj"],
        )

    def test_mlp_static_layers_do_not_disable_attention_row_routing(self) -> None:
        q_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        gate_up_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        modules = [
            ("model.layers.5.self_attn.q_proj", q_proj, q_proj.weight),
            (
                "model.layers.5.mlp.gate_up_proj",
                gate_up_proj,
                gate_up_proj.weight,
            ),
        ]
        env = {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": "test_model",
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": "/tmp/calibration",
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "full_sparse_residual",
            "SPECLINK_TOKEN_DENSE_MIXED_LAYERS": "all",
            "SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY": "all",
            "SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS": "5",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                speclink_structured_24,
                "_iter_target_modules",
                return_value=modules,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_load_activation_scales",
                return_value={},
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
            ),
            mock.patch.object(
                speclink_structured_24,
                "_attach_speclink_kernel_prepack",
            ),
        ):
            stats = speclink_structured_24.apply_structured_24_from_env(object())

        assert stats is not None
        self.assertTrue(q_proj._speclink_selective_mixed_rows)
        self.assertFalse(gate_up_proj._speclink_selective_mixed_rows)
        self.assertEqual(stats["speclink_kernel_mlp_static_layers"], [5])

    def test_pure_batch_routes_respect_mixed_projection_scope(self) -> None:
        q_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        gate_up_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
        modules = [
            ("model.layers.0.self_attn.q_proj", q_proj, q_proj.weight),
            (
                "model.layers.0.mlp.gate_up_proj",
                gate_up_proj,
                gate_up_proj.weight,
            ),
        ]
        env = {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": "test_model",
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": "/tmp/calibration",
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                "full_sparse_dense_override"
            ),
            "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": "batch_confidence",
            "SPECLINK_TOKEN_DENSE_MIXED_LAYERS": "all",
            "SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY": "gate_up",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                speclink_structured_24,
                "_iter_target_modules",
                return_value=modules,
            ),
            mock.patch.object(
                speclink_structured_24,
                "_load_activation_scales",
                return_value={},
            ),
            mock.patch.object(
                speclink_structured_24,
                "_import_speclink_kernel_backend",
            ),
            mock.patch.object(
                speclink_structured_24,
                "_attach_speclink_kernel_prepack",
            ),
        ):
            stats = speclink_structured_24.apply_structured_24_from_env(object())

        assert stats is not None
        self.assertTrue(stats["speclink_kernel_pure_batch_routes"])
        self.assertFalse(q_proj._speclink_selective_mixed_rows)
        self.assertTrue(gate_up_proj._speclink_selective_mixed_rows)
        self.assertEqual(
            stats["speclink_kernel_sparse_only_module_names"],
            ["model.layers.0.self_attn.q_proj"],
        )
        self.assertEqual(
            stats["speclink_kernel_mixed_module_names"],
            ["model.layers.0.mlp.gate_up_proj"],
        )

    def test_token_dense_bypass_module_uses_dense_dispatch(self) -> None:
        module = torch.nn.Linear(4, 3, bias=False)
        module._speclink_selective_dense_enabled = True
        module._speclink_selective_dense_bypass = True
        x = torch.randn(2, 4)

        with mock.patch.object(speclink_linear, "token_dense_enabled", return_value=True):
            actual = speclink_linear.speclink_linear_forward(module, x)

        expected = module(x)
        self.assertTrue(torch.equal(actual, expected))

    def test_unprepared_linear_bypasses_routing_context(self) -> None:
        module = torch.nn.Linear(4, 3, bias=False)
        x = torch.randn(2, 4)

        with (
            mock.patch.object(speclink_linear, "token_dense_enabled", return_value=True),
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                side_effect=AssertionError("routing context must not be read"),
            ),
        ):
            actual = speclink_linear.speclink_linear_forward(module, x)

        self.assertTrue(torch.equal(actual, module(x)))

    def test_auto_compile_dispatch_uses_prepacked_strategy(self) -> None:
        x = torch.randn(2, 4)
        for has_residual in (False, True):
            with self.subTest(has_residual=has_residual):
                module = SimpleNamespace(
                    weight=torch.randn(3, 4),
                    bias=None,
                    return_bias=False,
                    input_is_parallel=True,
                    _speclink_selective_dense_enabled=True,
                    _speclink_selective_mixed_rows=True,
                    _speclink_sparse24_full_a_values=torch.empty(3, 2),
                    _speclink_sparse24_full_a_meta_e=torch.empty(
                        1, dtype=torch.uint16
                    ),
                )
                if has_residual:
                    module._speclink_sparse24_residual_a_values = torch.empty(
                        3, 2
                    )
                    module._speclink_sparse24_residual_a_meta_e = torch.empty(
                        1, dtype=torch.uint16
                    )
                expected = torch.randn(2, 3)
                residual_result = expected if has_residual else torch.randn(2, 3)
                override_result = expected if not has_residual else torch.randn(2, 3)
                with (
                    mock.patch.object(
                        speclink_linear,
                        "token_dense_enabled",
                        return_value=True,
                    ),
                    mock.patch.object(
                        speclink_linear,
                        "linear_strategy",
                        return_value="auto",
                    ),
                    mock.patch.object(
                        torch.compiler,
                        "is_compiling",
                        return_value=True,
                    ),
                    mock.patch.object(
                        speclink_linear,
                        "current_verify_dense_row_summary",
                        side_effect=AssertionError(
                            "compile-safe auto dispatch must not read ContextVar"
                        ),
                    ),
                    mock.patch.object(
                        speclink_linear,
                        "_full_sparse_residual_compile_safe",
                        return_value=residual_result,
                    ) as residual,
                    mock.patch.object(
                        speclink_linear,
                        "_full_sparse_dense_override_compile_safe",
                        return_value=override_result,
                    ) as override,
                ):
                    actual = speclink_linear.speclink_linear_forward(module, x)

                self.assertIs(actual, expected)
                self.assertEqual(residual.call_count, int(has_residual))
                self.assertEqual(override.call_count, int(not has_residual))

    def test_unprepared_mlp_bypasses_routing_context(self) -> None:
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(),
            down_proj=SimpleNamespace(),
        )
        expected = torch.randn(2, 4)
        with (
            mock.patch.object(
                speclink_mlp,
                "_dense_mlp_forward",
                return_value=expected,
            ) as dense_forward,
        ):
            actual = speclink_mlp.speclink_mlp_forward(
                mlp,
                torch.randn(2, 4),
            )

        self.assertIs(actual, expected)
        dense_forward.assert_called_once()

    def test_routed_gate_dense_down_accepts_dense_down_bypass(self) -> None:
        gate_up = SimpleNamespace(
            _speclink_sparse24_routed_swiglu=True,
            _speclink_sparse24_gate_up_interleaved=True,
            _speclink_selective_dense_enabled=True,
            _speclink_selective_dense_bypass=False,
            _speclink_selective_mixed_rows=True,
            _speclink_gate_up_hybrid="none",
            _speclink_sparse24_full_a_values=torch.empty(8, 2),
            _speclink_sparse24_full_a_meta_e=torch.empty(
                1, dtype=torch.uint16
            ),
            _speclink_sparse24_residual_a_values=torch.empty(8, 2),
            _speclink_sparse24_residual_a_meta_e=torch.empty(
                1, dtype=torch.uint16
            ),
            tp_size=1,
            bias=None,
        )
        down = SimpleNamespace(
            _speclink_selective_dense_enabled=True,
            _speclink_selective_dense_bypass=True,
            _speclink_sparse24_dense_weight_released=False,
            tp_size=1,
            bias=None,
        )
        mlp = SimpleNamespace(
            gate_up_proj=gate_up,
            down_proj=down,
            act_fn=type("SiluAndMul", (), {})(),
        )
        x = SimpleNamespace(is_cuda=True, dtype=torch.float16, ndim=2)

        with (
            mock.patch.object(
                speclink_mlp, "_FUSED_BATCH_MLP_ENABLED", True
            ),
            mock.patch.object(
                speclink_mlp, "_ROUTED_SWIGLU_MLP_ENABLED", True
            ),
            mock.patch.object(
                speclink_mlp,
                "linear_strategy",
                return_value="full_sparse_residual",
            ),
            mock.patch.object(
                speclink_mlp, "mlp_strategy", return_value="linear"
            ),
        ):
            self.assertTrue(
                speclink_mlp._can_use_routed_gate_dense_down(mlp, x)
            )
            down._speclink_selective_dense_bypass = False
            self.assertFalse(
                speclink_mlp._can_use_routed_gate_dense_down(mlp, x)
            )

    def test_static_sparse_mlp_accepts_interleaved_gate_and_sparse_down(
        self,
    ) -> None:
        def sparse_module(**attributes: object) -> SimpleNamespace:
            return SimpleNamespace(
                _speclink_selective_dense_enabled=True,
                _speclink_selective_dense_bypass=False,
                _speclink_selective_mixed_rows=False,
                _speclink_sparse24_dense_weight_released=False,
                _speclink_sparse24_full_a_values=torch.empty(8, 2),
                _speclink_sparse24_full_a_meta_e=torch.empty(
                    1, dtype=torch.uint16
                ),
                tp_size=1,
                bias=None,
                **attributes,
            )

        gate_up = sparse_module(
            _speclink_sparse24_gate_up_interleaved=True,
            _speclink_sparse24_routed_swiglu=False,
            _speclink_gate_up_hybrid="none",
        )
        down = sparse_module()
        mlp = SimpleNamespace(
            gate_up_proj=gate_up,
            down_proj=down,
            act_fn=type("SiluAndMul", (), {})(),
        )
        x = SimpleNamespace(is_cuda=True, dtype=torch.float16, ndim=2)

        with (
            mock.patch.object(
                speclink_mlp, "_FUSED_BATCH_MLP_ENABLED", True
            ),
            mock.patch.object(
                speclink_mlp, "_INLINE_SWIGLU_MLP_ENABLED", True
            ),
            mock.patch.object(
                speclink_mlp, "_ROUTED_SWIGLU_MLP_ENABLED", False
            ),
            mock.patch.object(
                speclink_mlp,
                "linear_strategy",
                return_value="full_sparse_dense_override",
            ),
            mock.patch.object(
                speclink_mlp, "mlp_strategy", return_value="linear"
            ),
        ):
            self.assertTrue(
                speclink_mlp._can_use_static_sparse_mlp(mlp, x)
            )
            down._speclink_selective_mixed_rows = True
            self.assertFalse(
                speclink_mlp._can_use_static_sparse_mlp(mlp, x)
            )

    def test_static_sparse_mlp_dispatches_before_interleaved_guard(self) -> None:
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                _speclink_selective_dense_enabled=True,
                _speclink_selective_dense_bypass=False,
                _speclink_sparse24_gate_up_interleaved=True,
            ),
            down_proj=SimpleNamespace(
                _speclink_selective_dense_enabled=True,
                _speclink_selective_dense_bypass=False,
            ),
        )
        x = torch.randn(2, 4)
        expected = torch.randn(2, 4)
        with (
            mock.patch.object(
                speclink_mlp, "token_dense_enabled", return_value=True
            ),
            mock.patch.object(
                speclink_mlp, "_can_use_static_sparse_mlp", return_value=True
            ),
            mock.patch.object(
                speclink_mlp,
                "_static_sparse_mlp_forward",
                return_value=expected,
            ) as static_forward,
        ):
            actual = speclink_mlp.speclink_mlp_forward(mlp, x)

        self.assertIs(actual, expected)
        static_forward.assert_called_once_with(mlp, x)

    def test_fused_mixed_row_mlp_partitions_and_merges_once(self) -> None:
        x = torch.arange(16, dtype=torch.float32).view(4, 4)
        gate_weight = torch.arange(24, dtype=torch.float32).view(6, 4)
        down_weight = torch.arange(12, dtype=torch.float32).view(4, 3)
        dense_rows = torch.tensor([0, 2], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 3], dtype=torch.int32)
        sparse_gate = torch.randn(2, 6)
        sparse_hidden = torch.randn(2, 3)
        sparse_output = torch.randn(2, 4)

        def partition_rows(
            source: torch.Tensor,
            dense_indices: torch.Tensor,
            sparse_indices: torch.Tensor,
            dense_out: torch.Tensor,
            sparse_out: torch.Tensor,
        ) -> torch.Tensor:
            dense_out.copy_(source.index_select(0, dense_indices.long()))
            sparse_out.copy_(source.index_select(0, sparse_indices.long()))
            return source

        def merge_rows(
            output: torch.Tensor,
            dense_values: torch.Tensor,
            sparse_values: torch.Tensor,
            dense_indices: torch.Tensor,
            sparse_indices: torch.Tensor,
        ) -> torch.Tensor:
            output.index_copy_(0, dense_indices.long(), dense_values)
            output.index_copy_(0, sparse_indices.long(), sparse_values)
            return output

        with (
            mock.patch.object(
                speclink_mlp,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, True, False]),
                    2,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_mlp,
                "sparse24_partition_rows_",
                side_effect=partition_rows,
            ) as partition,
            mock.patch.object(
                speclink_mlp,
                "_silu_and_mul_contiguous",
                side_effect=lambda value: value[:, :3] + value[:, 3:],
            ),
            mock.patch.object(
                speclink_mlp,
                "sparse24_cutlass_device_gemm_prepacked",
                side_effect=(sparse_gate, sparse_output),
            ) as sparse_gemm,
            mock.patch.object(
                speclink_mlp,
                "sparse24_silu_and_mul_transposed",
                return_value=sparse_hidden,
            ),
            mock.patch.object(
                speclink_mlp,
                "sparse24_merge_rows_",
                side_effect=merge_rows,
            ) as merge,
        ):
            actual = speclink_mlp._batch_routed_mlp_impl(
                x,
                gate_weight,
                down_weight,
                torch.empty(6, 2),
                torch.empty(1, dtype=torch.uint16),
                torch.empty(4, 2),
                torch.empty(1, dtype=torch.uint16),
            )

        dense_x = x.index_select(0, dense_rows.long())
        dense_gate = dense_x @ gate_weight.t()
        dense_hidden = dense_gate[:, :3] + dense_gate[:, 3:]
        expected = torch.empty_like(actual)
        expected.index_copy_(0, dense_rows.long(), dense_hidden @ down_weight.t())
        expected.index_copy_(0, sparse_rows.long(), sparse_output)

        self.assertTrue(torch.equal(actual, expected))
        partition.assert_called_once()
        self.assertEqual(sparse_gemm.call_count, 2)
        merge.assert_called_once()

    def test_fused_residual_mlp_reconstructs_selected_rows(self) -> None:
        x = torch.arange(16, dtype=torch.float32).view(4, 4)
        dense_rows = torch.tensor([0, 2], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 3], dtype=torch.int32)
        full_gate = torch.randn(4, 6)
        residual_gate = torch.randn(2, 6)
        full_down = torch.randn(4, 4)
        residual_down = torch.randn(2, 4)

        def gather_rows(
            source: torch.Tensor,
            indices: torch.Tensor,
            out: torch.Tensor,
        ) -> torch.Tensor:
            out.copy_(source.index_select(0, indices.long()))
            return out

        def add_rows(
            output: torch.Tensor,
            values: torch.Tensor,
            indices: torch.Tensor,
        ) -> torch.Tensor:
            output.index_add_(0, indices.long(), values)
            return output

        with (
            mock.patch.object(
                speclink_mlp,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, True, False]),
                    2,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_mlp,
                "sparse24_gather_rows_",
                side_effect=gather_rows,
            ) as gather,
            mock.patch.object(
                speclink_mlp,
                "_silu_and_mul_contiguous",
                side_effect=lambda value: value[:, :3] + value[:, 3:],
            ),
            mock.patch.object(
                speclink_mlp,
                "sparse24_cutlass_device_gemm_prepacked",
                side_effect=(
                    full_gate.clone(),
                    residual_gate,
                    full_down.clone(),
                    residual_down,
                ),
            ) as sparse_gemm,
            mock.patch.object(
                speclink_mlp,
                "sparse24_add_indexed_rows_contiguous_",
                side_effect=add_rows,
            ) as add,
        ):
            actual = speclink_mlp._batch_routed_residual_mlp_impl(
                x,
                torch.empty(6, 2),
                torch.empty(1, dtype=torch.uint16),
                torch.empty(6, 2),
                torch.empty(1, dtype=torch.uint16),
                torch.empty(4, 2),
                torch.empty(1, dtype=torch.uint16),
                torch.empty(4, 2),
                torch.empty(1, dtype=torch.uint16),
            )

        corrected_gate = full_gate.clone()
        corrected_gate.index_add_(0, dense_rows.long(), residual_gate)
        expected_hidden = corrected_gate[:, :3] + corrected_gate[:, 3:]
        expected = full_down.clone()
        expected.index_add_(0, dense_rows.long(), residual_down)

        self.assertTrue(
            torch.equal(gather.call_args_list[1].args[0], expected_hidden)
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(sparse_gemm.call_count, 4)
        self.assertEqual(add.call_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_residual_mlp_matches_dense_and_graph_replay_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(20260714)
        rows = 16
        hidden_size = 64
        intermediate_size = 256
        gate_weight = torch.randn(
            (hidden_size, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        down_weight = torch.randn(
            (intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)

        gate24, _ = apply_random_24_mask(gate_weight, generator=generator)
        gate_residual24 = gate_weight - gate24
        gate_packed = pack_24(gate24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=hidden_size,
        )
        gate_residual_packed = pack_24(
            gate_residual24, layout="n_major"
        )
        gate_residual_values, gate_residual_meta = (
            prepare_cutlass_sparse24_device_gemm(
                gate_residual_packed.values,
                gate_residual_packed.meta,
                layout=gate_residual_packed.layout,
                K=hidden_size,
            )
        )

        down24, _ = apply_random_24_mask(down_weight, generator=generator)
        down_residual24 = down_weight - down24
        down_packed = pack_24(down24, layout="n_major")
        down_values, down_meta = prepare_cutlass_sparse24_device_gemm(
            down_packed.values,
            down_packed.meta,
            layout=down_packed.layout,
            K=intermediate_size,
        )
        down_residual_packed = pack_24(
            down_residual24, layout="n_major"
        )
        down_residual_values, down_residual_meta = (
            prepare_cutlass_sparse24_device_gemm(
                down_residual_packed.values,
                down_residual_packed.meta,
                layout=down_residual_packed.layout,
                K=intermediate_size,
            )
        )

        dense_mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
        dense_mask[torch.tensor([0, 3, 6, 9, 12], device="cuda")] = True
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=5,
            sparse_count=11,
            total_rows=rows,
        )
        x = torch.randn(
            (rows, hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.1)

        def dense_reference(value: torch.Tensor) -> torch.Tensor:
            gate_up = value @ gate_weight
            hidden = torch.empty(
                (rows, intermediate_size),
                device="cuda",
                dtype=torch.float16,
            )
            torch.ops._C.silu_and_mul(hidden, gate_up)
            return hidden @ down_weight

        def routed(value: torch.Tensor) -> torch.Tensor:
            return speclink_mlp._batch_routed_residual_mlp_impl(
                value,
                gate_values,
                gate_meta,
                gate_residual_values,
                gate_residual_meta,
                down_values,
                down_meta,
                down_residual_values,
                down_residual_meta,
            )

        speclink_mlp.prepare_mixed_mlp_streams(torch.device("cuda"))
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(speclink_mlp, "_ROUTED_SWIGLU_MLP_ENABLED", True),
            mock.patch.object(
                speclink_mlp,
                "_DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED",
                False,
            ),
        ):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                expected = dense_reference(x)
                actual = routed(x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=5e-2, atol=2e-1)
                )

                static_x = x.clone()
                for _ in range(3):
                    graph_output = routed(static_x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = routed(static_x)
                replay_x = torch.randn_like(static_x).mul_(0.1)
                static_x.copy_(replay_x)
                graph.replay()
                torch.cuda.synchronize()
                replay_expected = dense_reference(replay_x)
                self.assertTrue(
                    torch.allclose(
                        graph_output,
                        replay_expected,
                        rtol=5e-2,
                        atol=2e-1,
                    )
                )
            finally:
                speclink_token_dense.end_verify_context(token)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_gate_dense_down_hidden_tracks_dynamic_graph_route_cuda(
        self,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(20260715)
        rows = 48
        hidden_size = 64
        intermediate_size = 256
        gate_weight = torch.randn(
            (hidden_size, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        gate24, _ = apply_random_24_mask(gate_weight, generator=generator)
        gate_packed = pack_24(gate24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=hidden_size,
        )
        residual_packed = pack_24(
            gate_weight - gate24, layout="n_major"
        )
        residual_values, residual_meta = (
            prepare_cutlass_sparse24_device_gemm(
                residual_packed.values,
                residual_packed.meta,
                layout=residual_packed.layout,
                K=hidden_size,
            )
        )
        x = torch.randn(
            (rows, hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.1)

        def make_plan(sparse_rows: list[int]):
            dense_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
            dense_mask[torch.tensor(sparse_rows, device="cuda")] = False
            plan = speclink_token_dense._make_plan(
                dense_mask,
                dense_count=rows - len(sparse_rows),
                sparse_count=len(sparse_rows),
                total_rows=rows,
            )
            return speclink_token_dense.pad_verify_plan_for_cudagraph(
                plan,
                actual_rows=rows,
                padded_rows=rows,
                device=torch.device("cuda"),
            )

        def run(value: torch.Tensor) -> torch.Tensor:
            return torch.ops.speclink.routed_residual_gate_swiglu.default(
                value,
                gate_values,
                gate_meta,
                residual_values,
                residual_meta,
            )

        def reference(
            value: torch.Tensor, sparse_rows: list[int]
        ) -> torch.Tensor:
            gate_up = value @ gate_weight
            if sparse_rows:
                sparse_ids = torch.tensor(sparse_rows, device="cuda")
                gate_up[sparse_ids] = value[sparse_ids] @ gate24
            hidden = torch.empty(
                (rows, intermediate_size),
                device="cuda",
                dtype=torch.float16,
            )
            torch.ops._C.silu_and_mul(hidden, gate_up)
            return hidden

        speclink_mlp.prepare_mixed_mlp_streams(torch.device("cuda"))
        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(
                speclink_token_dense, "_STATIC_GRAPH_ROUTING", True
            ),
            mock.patch.object(
                speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8
            ),
            mock.patch.object(
                speclink_token_dense, "_STATIC_SPARSE_BONUS", False
            ),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_DENSE_SELECTION",
                "global_topk",
            ),
            mock.patch.object(
                speclink_mlp, "_ROUTED_SWIGLU_MLP_ENABLED", True
            ),
            mock.patch.object(
                speclink_mlp,
                "_DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED",
                False,
            ),
            mock.patch.dict(
                os.environ,
                {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "32"},
            ),
        ):
            capture_sparse_rows = list(range(40, 48))
            capture_plan = make_plan(capture_sparse_rows)
            assert capture_plan is not None
            warmup_token = speclink_token_dense.begin_verify_context(
                capture_plan
            )
            eager_output = run(x)
            speclink_token_dense.end_verify_context(warmup_token)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(
                    eager_output,
                    reference(x, capture_sparse_rows),
                    rtol=5e-2,
                    atol=2e-1,
                )
            )

            capture_token = speclink_token_dense.begin_verify_context(
                capture_plan
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = run(x)
            speclink_token_dense.end_verify_context(capture_token)

            runtime_sparse_rows = list(range(8))
            runtime_plan = make_plan(runtime_sparse_rows)
            assert runtime_plan is not None
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(
                    graph_output,
                    reference(x, runtime_sparse_rows),
                    rtol=5e-2,
                    atol=2e-1,
                ),
                "CUDA graph replay did not consume the updated Gate row route",
            )

            all_sparse_mask = torch.zeros(
                rows, device="cuda", dtype=torch.bool
            )
            all_sparse_plan = speclink_token_dense._make_plan(
                all_sparse_mask,
                dense_count=0,
                sparse_count=rows,
                total_rows=rows,
            )
            all_sparse_token = speclink_token_dense.begin_verify_context(
                all_sparse_plan
            )
            all_sparse_output = run(x)
            speclink_token_dense.end_verify_context(all_sparse_token)
            self.assertTrue(
                torch.allclose(
                    all_sparse_output,
                    reference(x, list(range(rows))),
                    rtol=5e-2,
                    atol=2e-1,
                )
            )

        with mock.patch.object(
            speclink_mlp,
            "_DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED",
            False,
        ):
            dense_output = run(x)
            self.assertTrue(
                torch.allclose(
                    dense_output,
                    reference(x, []),
                    rtol=5e-2,
                    atol=2e-1,
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sparse_gate_dense_down_matches_route_and_graph_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(20260714)
        rows = 16
        hidden_size = 64
        intermediate_size = 256
        gate_weight = torch.randn(
            (hidden_size, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        gate24, _ = apply_random_24_mask(gate_weight, generator=generator)
        gate_packed = pack_24(gate24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=hidden_size,
        )
        down_weight = torch.randn(
            (intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        x = torch.randn(
            (rows, hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.1)

        def reference(
            value: torch.Tensor, gate: torch.Tensor
        ) -> torch.Tensor:
            gate_up = value @ gate
            hidden = torch.empty(
                (value.shape[0], intermediate_size),
                device="cuda",
                dtype=torch.float16,
            )
            torch.ops._C.silu_and_mul(hidden, gate_up)
            return hidden @ down_weight

        def run(value: torch.Tensor) -> torch.Tensor:
            hidden = torch.ops.speclink.sparse_gate_swiglu.default(
                value,
                gate_weight.t().contiguous(),
                gate_values,
                gate_meta,
            )
            return hidden @ down_weight

        dense_mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=0,
            sparse_count=rows,
            total_rows=rows,
        )
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                expected = reference(x, gate24)
                actual = run(x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=5e-2, atol=2e-1)
                )

                static_x = x.clone()
                for _ in range(3):
                    graph_output = run(static_x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = run(static_x)
                replay_x = torch.randn_like(static_x).mul_(0.1)
                static_x.copy_(replay_x)
                graph.replay()
                torch.cuda.synchronize()
                replay_expected = reference(replay_x, gate24)
                self.assertTrue(
                    torch.allclose(
                        graph_output,
                        replay_expected,
                        rtol=5e-2,
                        atol=2e-1,
                    )
                )
            finally:
                speclink_token_dense.end_verify_context(token)

        prefill_count = 4
        speclink_mlp.prepare_mixed_mlp_streams(torch.device("cuda"))
        prefill_mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
        prefill_mask[:prefill_count] = True
        mixed_plan = speclink_token_dense._make_plan(
            prefill_mask,
            dense_count=prefill_count,
            sparse_count=rows - prefill_count,
            total_rows=rows,
            has_prefill_rows=True,
            prefill_mask=prefill_mask,
            contiguous_prefill_prefix=True,
        )
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            token = speclink_token_dense.begin_verify_context(mixed_plan)
            try:
                mixed_actual = run(x)
                mixed_expected = torch.cat(
                    (
                        reference(x[:prefill_count], gate_weight),
                        reference(x[prefill_count:], gate24),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        mixed_actual,
                        mixed_expected,
                        rtol=5e-2,
                        atol=2e-1,
                    )
                )
            finally:
                speclink_token_dense.end_verify_context(token)

        dense_actual = run(x)
        dense_expected = reference(x, gate_weight)
        self.assertTrue(
            torch.allclose(
                dense_actual,
                dense_expected,
                rtol=5e-2,
                atol=2e-1,
            )
        )

    def test_sparse_only_custom_op_impl_uses_dense_fallback(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        with mock.patch.object(
            speclink_linear,
            "current_verify_dense_row_summary",
            return_value=None,
        ):
            actual = speclink_linear._sparse_only_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
                True,
            )

        self.assertTrue(torch.equal(actual, x @ weight.t()))

        view = speclink_linear._sparse_only_linear_impl(
            x,
            weight,
            torch.empty(5, 2),
            torch.empty(1, dtype=torch.uint16),
            False,
        )
        self.assertTrue(torch.equal(view, x @ weight.t()))
        self.assertEqual(view.stride(), (1, 8))
        fake = speclink_linear._sparse_only_linear_fake(
            x,
            weight,
            torch.empty(5, 2),
            torch.empty(1, dtype=torch.uint16),
            False,
        )
        self.assertEqual(fake.stride(), (1, 8))

    def test_sparse_only_custom_op_impl_accepts_zero_dense_plan(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        expected = torch.randn(3, 5)
        empty_rows = torch.empty(0, dtype=torch.int32)
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.zeros(3, dtype=torch.bool),
                    0,
                    empty_rows,
                    torch.arange(3, dtype=torch.int32),
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
                return_value=expected,
            ) as sparse_gemm,
        ):
            actual = speclink_linear._sparse_only_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
                True,
            )

        self.assertIs(actual, expected)
        sparse_gemm.assert_called_once()

    def test_force_sparse_gate_up_can_use_direct_store_epilogue(self) -> None:
        x = torch.randn(16, 4096)
        weight = torch.empty((24576, 4096), device="meta")
        expected = torch.randn(16, 24576)
        sparse_values = torch.empty((24576, 1), device="meta")
        sparse_meta = torch.empty(1, dtype=torch.uint16, device="meta")
        empty_rows = torch.empty(0, dtype=torch.int32)
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.zeros(16, dtype=torch.bool),
                    0,
                    empty_rows,
                    torch.arange(16, dtype=torch.int32),
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "_DIRECT_STORE_GATE_UP_ENABLED",
                True,
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_inline_transpose_gemm_prepacked",
                return_value=expected,
            ) as direct_store,
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
            ) as sparse_gemm,
        ):
            actual = speclink_linear._force_sparse_linear_impl(
                x,
                weight,
                sparse_values,
                sparse_meta,
                True,
            )

        self.assertIs(actual, expected)
        direct_store.assert_called_once_with(
            x,
            sparse_values,
            sparse_meta,
            config="auto",
            store_mode="vector",
        )
        sparse_gemm.assert_not_called()

    def test_force_sparse_custom_op_ignores_mixed_dense_rows(self) -> None:
        x = torch.randn(4, 8)
        weight = torch.randn(6, 8)
        expected = torch.randn(4, 6)
        dense_rows = torch.tensor([0, 3], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 2], dtype=torch.int32)
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, False, True]),
                    2,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
                return_value=expected,
            ) as sparse_gemm,
        ):
            actual = speclink_linear._force_sparse_linear_impl(
                x,
                weight,
                torch.empty(6, 4),
                torch.empty(1, dtype=torch.uint16),
                True,
            )

        self.assertIs(actual, expected)
        sparse_gemm.assert_called_once()

    def test_force_sparse_custom_op_splits_mixed_prefill_decode_rows(self) -> None:
        x = torch.randn(4, 8)
        weight = torch.randn(6, 8)
        sparse_weight = weight * torch.tensor(
            [1, 1, 0, 0, 1, 1, 0, 0],
            dtype=weight.dtype,
        )
        dense_mask = torch.tensor([True, False, True, True])
        prefill_mask = torch.tensor([False, False, True, True])
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=3,
            sparse_count=1,
            total_rows=4,
            has_prefill_rows=True,
            prefill_mask=prefill_mask,
            contiguous_prefill_suffix=True,
        )

        def copy_rows(
            output: torch.Tensor,
            values: torch.Tensor,
            rows: torch.Tensor,
        ) -> torch.Tensor:
            output.index_copy_(0, rows.long(), values)
            return output

        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(
                speclink_linear,
                "_gather_rows",
                side_effect=lambda tensor, rows: tensor.index_select(
                    0, rows.long()
                ),
            ) as gather_rows,
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
                side_effect=lambda tensor, *_args, **_kwargs: (
                    tensor @ sparse_weight.t()
                ),
            ) as sparse_gemm,
            mock.patch.object(
                speclink_linear,
                "sparse24_copy_indexed_rows_contiguous_",
                side_effect=copy_rows,
            ),
        ):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                actual = speclink_linear._force_sparse_linear_impl(
                    x,
                    weight,
                    torch.empty(6, 4),
                    torch.empty(1, dtype=torch.uint16),
                    True,
                )
            finally:
                speclink_token_dense.end_verify_context(token)

        expected = x @ sparse_weight.t()
        expected[2:] = x[2:] @ weight.t()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[2:], x[2:] @ weight.t()))
        self.assertTrue(torch.equal(sparse_gemm.call_args.args[0], x[:2]))
        gather_rows.assert_not_called()

    def test_mixed_override_custom_op_impl_uses_dense_fallback(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        with mock.patch.object(
            speclink_linear,
            "current_verify_dense_row_summary",
            return_value=None,
        ):
            actual = speclink_linear._mixed_dense_override_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
            )

        self.assertTrue(torch.equal(actual, x @ weight.t()))

    def test_mixed_override_custom_op_impl_uses_fixed_dense_rows(self) -> None:
        x = torch.randn(4, 8)
        weight = torch.randn(6, 8)
        dense_rows = torch.tensor([0, 3], dtype=torch.int32)
        expected = torch.randn(4, 6)
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, False, True]),
                    2,
                    dense_rows,
                    torch.tensor([1, 2], dtype=torch.int32),
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_mixed_dense_override_prepacked",
                return_value=expected,
            ) as mixed_gemm,
        ):
            actual = speclink_linear._mixed_dense_override_linear_impl(
                x,
                weight,
                torch.empty(6, 4),
                torch.empty(1, dtype=torch.uint16),
            )

        self.assertIs(actual, expected)
        self.assertTrue(torch.equal(mixed_gemm.call_args.args[4], dense_rows))
        fake = speclink_linear._mixed_dense_override_linear_fake(
            x,
            weight,
            torch.empty(6, 4),
            torch.empty(1, dtype=torch.uint16),
        )
        self.assertEqual(fake.shape, (4, 6))

    def test_split_dense_sparse_custom_op_uses_dense_fallback(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        with mock.patch.object(
            speclink_linear,
            "current_verify_dense_row_summary",
            return_value=None,
        ):
            actual = speclink_linear._split_dense_sparse_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
            )

        self.assertTrue(torch.equal(actual, x @ weight.t()))
        fake = speclink_linear._split_dense_sparse_linear_fake(
            x,
            weight,
            torch.empty(5, 2),
            torch.empty(1, dtype=torch.uint16),
        )
        self.assertEqual(fake.shape, (3, 5))

    def test_sparse_only_custom_op_preserves_transposed_view(self) -> None:
        x = torch.randn(4, 8).t()
        weight = torch.randn(5, 4)
        expected = torch.empty_strided((8, 5), (1, 8))
        empty_rows = torch.empty(0, dtype=torch.int32)
        with (
            mock.patch.dict(
                os.environ,
                {"SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT": "1"},
                clear=True,
            ),
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.zeros(8, dtype=torch.bool),
                    0,
                    empty_rows,
                    torch.arange(8, dtype=torch.int32),
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
                return_value=expected,
            ) as sparse_gemm,
        ):
            actual = speclink_linear._sparse_only_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
                False,
            )

        self.assertIs(actual, expected)
        self.assertEqual(actual.stride(), (1, 8))
        self.assertIs(sparse_gemm.call_args.args[0], x)
        self.assertFalse(sparse_gemm.call_args.kwargs["contiguous_output"])
        self.assertTrue(sparse_gemm.call_args.kwargs["input_transposed"])

        fake = speclink_linear._sparse_only_linear_fake(
            x,
            weight,
            torch.empty(5, 2),
            torch.empty(1, dtype=torch.uint16),
            False,
        )
        self.assertEqual(fake.stride(), (1, 8))

    def test_full_sparse_residual_custom_op_impl_uses_dense_fallback(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        with mock.patch.object(
            speclink_linear,
            "current_verify_dense_row_summary",
            return_value=None,
        ):
            actual = speclink_linear._full_sparse_residual_linear_impl(
                x,
                weight,
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
                torch.empty(5, 2),
                torch.empty(1, dtype=torch.uint16),
            )

        self.assertTrue(torch.equal(actual, x @ weight.t()))

    def test_full_sparse_residual_prefers_qkv_heterogeneous_kernel(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        full_values = torch.empty(5, 2)
        full_meta = torch.empty(1, dtype=torch.uint16)
        residual_values = torch.empty(5, 2)
        residual_meta = torch.empty(1, dtype=torch.uint16)
        dense_rows = torch.tensor([0], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 2], dtype=torch.int32)
        expected = torch.randn(3, 5)

        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, False]),
                    1,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "_qkv_heterogeneous_exact",
                return_value=expected,
            ) as heterogeneous,
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
            ) as sparse_gemm,
        ):
            actual = speclink_linear._full_sparse_residual_linear_impl(
                x,
                weight,
                full_values,
                full_meta,
                residual_values,
                residual_meta,
            )

        self.assertIs(actual, expected)
        heterogeneous.assert_called_once_with(
            x,
            weight,
            full_values,
            full_meta,
            dense_rows,
            sparse_rows,
        )
        sparse_gemm.assert_not_called()

    def test_full_sparse_residual_prefers_profiled_qkv_paired_kernel(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        full_values = torch.empty(5, 2)
        full_meta = torch.empty(1, dtype=torch.uint16)
        residual_values = torch.empty(5, 2)
        residual_meta = torch.empty(1, dtype=torch.uint16)
        dense_rows = torch.tensor([0], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 2], dtype=torch.int32)
        expected = torch.randn(3, 5)

        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, False]),
                    1,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "_qkv_paired_exact",
                return_value=expected,
            ) as paired,
            mock.patch.object(
                speclink_linear,
                "_qkv_heterogeneous_exact",
            ) as heterogeneous,
        ):
            actual = speclink_linear._full_sparse_residual_linear_impl(
                x,
                weight,
                full_values,
                full_meta,
                residual_values,
                residual_meta,
            )

        self.assertIs(actual, expected)
        paired.assert_called_once_with(
            x,
            weight,
            full_values,
            full_meta,
            residual_values,
            residual_meta,
            dense_rows,
        )
        heterogeneous.assert_not_called()

    def test_qkv_paired_selector_uses_profiled_qkv_shapes(self) -> None:
        with mock.patch.object(
            speclink_linear,
            "_QKV_ACTIVE_WAVE_C12_ENABLED",
            True,
        ):
            self.assertTrue(
                speclink_linear._qkv_use_paired_backend(112, 30, 6144)
            )
            self.assertTrue(
                speclink_linear._qkv_use_paired_backend(144, 40, 6144)
            )
            self.assertTrue(
                speclink_linear._qkv_use_paired_backend(180, 20, 6144)
            )
            self.assertFalse(
                speclink_linear._qkv_use_paired_backend(168, 45, 6144)
            )
            self.assertTrue(
                speclink_linear._qkv_use_paired_backend(196, 28, 6144)
            )
            self.assertTrue(
                speclink_linear._qkv_use_paired_backend(288, 32, 6144)
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(112, 30, 6144),
                "256x32_full_256x32_residual_contiguous",
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(224, 60, 6144),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(224, 32, 6144),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(576, 64, 6144),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(448, 32, 6144),
                "256x64_full_256x64_residual_contiguous",
            )
            self.assertIsNone(
                speclink_linear._qkv_paired_config(705, 32, 6144)
            )

        with mock.patch.object(
            speclink_linear,
            "_QKV_ACTIVE_WAVE_C12_ENABLED",
            False,
        ):
            self.assertFalse(
                speclink_linear._qkv_use_paired_backend(196, 28, 6144)
            )
            self.assertEqual(
                speclink_linear._qkv_paired_config(224, 32, 6144),
                "256x64_full_256x32_residual_contiguous",
            )
            self.assertIsNone(
                speclink_linear._qkv_paired_config(448, 32, 6144)
            )

    def test_qkv_paired_exact_accepts_released_dense_weight(self) -> None:
        x = mock.Mock()
        x.shape = (144, 4096)
        x.is_cuda = True
        x.dtype = torch.float16
        x.device = torch.device("cuda")
        x.is_contiguous.return_value = True
        dense_weight = torch.empty(0)
        full_values = mock.Mock()
        full_values.shape = (6144, 2048)
        dense_rows = mock.Mock()
        dense_rows.numel.return_value = 16
        full_buffer = torch.empty((144, 6144))

        with (
            mock.patch.object(
                speclink_linear,
                "_QKV_PAIRED_ROUTING_ENABLED",
                True,
            ),
            mock.patch.object(
                speclink_linear,
                "_QKV_PAIRED_MAX_ROWS",
                704,
            ),
            mock.patch.object(
                speclink_linear,
                "current_verify_prefill_row_summary",
                return_value=None,
            ),
            mock.patch.object(
                speclink_linear,
                "_cuda_graph_capturing",
                return_value=False,
            ),
            mock.patch.object(
                speclink_linear,
                "_cached_sparse_buffers",
                return_value=(full_buffer, mock.Mock()),
            ),
            mock.patch.object(
                speclink_linear,
                "_cached_qkv_paired_residual",
                return_value=mock.Mock(),
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_paired_gather_residual_prepacked",
            ) as paired,
            mock.patch.object(
                speclink_linear,
                "sparse24_add_indexed_rows_contiguous_",
            ),
        ):
            actual = speclink_linear._qkv_paired_exact(
                x,
                dense_weight,
                full_values,
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                dense_rows,
            )

        self.assertEqual(tuple(actual.shape), (144, 6144))
        self.assertEqual(actual.data_ptr(), full_buffer.data_ptr())
        paired.assert_called_once()

    def test_fused_override_mlp_backend_uses_profiled_row_ranges(self) -> None:
        self.assertEqual(
            speclink_mlp._fused_override_backend(224, 32, 24576),
            "heterogeneous",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(288, 32, 24576),
            "persistent_gate",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(576, 64, 24576),
            "persistent_gate",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(616, 70, 24576),
            "parallel_gate",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(352, 100, 28672),
            "parallel_gate",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(440, 125, 28672),
            "persistent_gate",
        )
        self.assertEqual(
            speclink_mlp._fused_override_backend(504, 140, 28672),
            "parallel_gate",
        )
        self.assertIsNone(
            speclink_mlp._fused_override_backend(252, 28, 24576)
        )

    def test_mixed_dense_override_prefers_qkv_heterogeneous_kernel(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        full_values = torch.empty(5, 2)
        full_meta = torch.empty(1, dtype=torch.uint16)
        dense_rows = torch.tensor([0], dtype=torch.int32)
        sparse_rows = torch.tensor([1, 2], dtype=torch.int32)
        expected = torch.randn(3, 5)

        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.tensor([True, False, False]),
                    1,
                    dense_rows,
                    sparse_rows,
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "_qkv_heterogeneous_exact",
                return_value=expected,
            ) as heterogeneous,
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
            ) as sparse_gemm,
        ):
            actual = speclink_linear._mixed_dense_override_linear_impl(
                x,
                weight,
                full_values,
                full_meta,
            )

        self.assertIs(actual, expected)
        heterogeneous.assert_called_once_with(
            x,
            weight,
            full_values,
            full_meta,
            dense_rows,
            sparse_rows,
        )
        sparse_gemm.assert_not_called()

    def test_force_sparse_released_weight_uses_context_aware_custom_op(self) -> None:
        x = torch.randn(3, 4)
        expected = torch.randn(3, 5)
        module = SimpleNamespace(
            weight=torch.empty(0),
            _speclink_sparse24_dense_weight_released=True,
            _speclink_sparse24_full_a_values=torch.empty(5, 2),
            _speclink_sparse24_full_a_meta_e=torch.empty(
                1, dtype=torch.uint16
            ),
            _speclink_sparse24_residual_a_values=torch.empty(5, 2),
            _speclink_sparse24_residual_a_meta_e=torch.empty(
                1, dtype=torch.uint16
            ),
        )
        residual_op = torch.ops.speclink.released_force_sparse_linear

        with mock.patch.object(
            residual_op,
            "default",
            return_value=expected,
        ) as residual:
            actual = speclink_linear._force_sparse_decode_compile_safe(
                module,
                x,
            )

        self.assertIs(actual, expected)
        residual.assert_called_once_with(
            x,
            module.weight,
            module._speclink_sparse24_full_a_values,
            module._speclink_sparse24_full_a_meta_e,
            module._speclink_sparse24_residual_a_values,
            module._speclink_sparse24_residual_a_meta_e,
            True,
        )

    def test_released_static_projection_uses_only_w24_during_verify(self) -> None:
        x = torch.randn(3, 4)
        full_values = torch.empty(5, 2)
        full_meta = torch.empty(1, dtype=torch.uint16)
        residual_values = torch.empty(5, 2)
        residual_meta = torch.empty(1, dtype=torch.uint16)
        expected = torch.randn(3, 5)
        empty_rows = torch.empty(0, dtype=torch.int32)

        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=(
                    torch.ones(3, dtype=torch.bool),
                    3,
                    torch.arange(3, dtype=torch.int32),
                    empty_rows,
                ),
            ),
            mock.patch.object(
                speclink_linear,
                "current_verify_prefill_row_summary",
                return_value=None,
            ),
            mock.patch.object(
                speclink_linear,
                "sparse24_cutlass_device_gemm_prepacked",
                return_value=expected,
            ) as sparse_gemm,
            mock.patch.object(
                speclink_linear,
                "_full_sparse_residual_linear_core",
            ) as exact_fallback,
        ):
            actual = speclink_linear._released_force_sparse_linear_impl(
                x,
                torch.empty(0),
                full_values,
                full_meta,
                residual_values,
                residual_meta,
                True,
            )

        self.assertIs(actual, expected)
        sparse_gemm.assert_called_once_with(
            x.contiguous(),
            full_values,
            full_meta,
            contiguous_output=True,
        )
        exact_fallback.assert_not_called()

    def test_sparse_only_mlp_bypasses_routing_context(self) -> None:
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(_speclink_selective_dense_enabled=True),
            down_proj=SimpleNamespace(_speclink_selective_dense_enabled=True),
        )
        expected = torch.randn(2, 4)
        with (
            mock.patch.object(
                speclink_mlp,
                "token_dense_enabled",
                return_value=True,
            ),
            mock.patch.object(
                speclink_mlp,
                "mlp_strategy",
                return_value="linear",
            ),
            mock.patch.object(
                speclink_mlp,
                "_linear_mlp_forward",
                return_value=expected,
            ) as linear_forward,
        ):
            actual = speclink_mlp.speclink_mlp_forward(
                mlp,
                torch.randn(2, 4),
            )

            self.assertIs(actual, expected)
            linear_forward.assert_called_once()

    def test_linear_mlp_uses_layout_aware_gate_activation(self) -> None:
        x = torch.randn(2, 4)
        gate_up = torch.randn(2, 8)
        hidden = torch.randn(2, 4)
        expected = torch.randn(2, 4)
        mlp = SimpleNamespace(
            gate_up_proj=object(),
            down_proj=object(),
            act_fn=mock.Mock(
                side_effect=AssertionError(
                    "the contiguous activation must not read CUTLASS layout"
                )
            ),
        )

        with (
            mock.patch.object(
                speclink_mlp,
                "speclink_linear_forward",
                side_effect=((gate_up, None), (expected, None)),
            ) as linear_forward,
            mock.patch.object(
                speclink_mlp,
                "_is_cutlass_transposed_gate_up",
                return_value=True,
            ),
            mock.patch.object(
                torch.ops.speclink.transposed_silu_and_mul_contiguous,
                "default",
                return_value=hidden,
            ) as transposed_activation,
        ):
            actual = speclink_mlp._linear_mlp_forward(mlp, x)

        self.assertIs(actual, expected)
        transposed_activation.assert_called_once_with(gate_up)
        self.assertEqual(linear_forward.call_args_list[1].args, (mlp.down_proj, hidden))

    def test_all_dense_mlp_bypasses_routing_context(self) -> None:
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                _speclink_selective_dense_bypass=True
            ),
            down_proj=SimpleNamespace(
                _speclink_selective_dense_bypass=True
            ),
        )
        expected = torch.randn(2, 4)
        with (
            mock.patch.object(
                speclink_mlp,
                "_dense_mlp_forward",
                return_value=expected,
            ) as dense_forward,
        ):
            actual = speclink_mlp.speclink_mlp_forward(
                mlp,
                torch.randn(2, 4),
            )

        self.assertIs(actual, expected)
        dense_forward.assert_called_once()

    def test_prefill_forced_dense_consumes_pending_scores(self) -> None:
        speclink_token_dense._pending_scores["req0"].append([0.2, 0.1])
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "16",
                },
            ):
                mask = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["req0"],
                    num_scheduled_tokens=[4],
                    num_draft_tokens=[2],
                    num_decode_draft_tokens=[-1],
                    cu_num_scheduled_tokens=[4],
                    total_num_scheduled_tokens=4,
                    device=torch.device("cpu"),
                )

        self.assertIsNone(mask)
        self.assertEqual(len(speclink_token_dense._pending_scores["req0"]), 0)

    def test_sparse_unscored_decode_keeps_only_prefill_dense(self) -> None:
        speclink_token_dense._pending_scores["draft"].append([0.9, 0.8])
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0",
                    "SPECLINK_TOKEN_DENSE_SPARSE_BONUS": "1",
                    "SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE": "1",
                },
            ):
                plan = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["draft", "fallback", "prefill"],
                    num_scheduled_tokens=[3, 1, 3],
                    num_draft_tokens=[2, 0, 0],
                    num_decode_draft_tokens=[2, 0, -1],
                    cu_num_scheduled_tokens=[3, 4, 7],
                    total_num_scheduled_tokens=7,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [False, False, False, False, True, True, True],
        )
        self.assertEqual(plan.dense_count, 3)
        self.assertEqual(plan.sparse_count, 4)
        self.assertTrue(plan.has_prefill_rows)
        self.assertEqual(plan.prefill_count, 3)
        self.assertEqual(plan.decode_count, 4)
        self.assertEqual(plan.prefill_rows[:3].tolist(), [4, 5, 6])
        self.assertEqual(plan.decode_rows[:4].tolist(), [0, 1, 2, 3])
        self.assertTrue(plan.contiguous_prefill_suffix)
        self.assertFalse(plan.contiguous_prefill_prefix)

    def test_sparse_unscored_decode_routes_missing_scores_sparse(self) -> None:
        speclink_token_dense._pending_scores["draft"].append([0.9])
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0",
                    "SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE": "1",
                },
            ):
                plan = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["draft"],
                    num_scheduled_tokens=[4],
                    num_draft_tokens=[3],
                    num_decode_draft_tokens=[3],
                    cu_num_scheduled_tokens=[4],
                    total_num_scheduled_tokens=4,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.dense_mask.tolist(), [False, False, False, True])
        self.assertEqual(plan.dense_count, 1)
        self.assertEqual(plan.sparse_count, 3)

    def test_sparse_only_decode_builds_all_sparse_plan_without_scores(self) -> None:
        speclink_token_dense._pending_scores["req0"].append([0.2, 0.1])
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0",
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
                },
            ):
                plan = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["req0"],
                    num_scheduled_tokens=[4],
                    num_draft_tokens=[2],
                    num_decode_draft_tokens=[2],
                    cu_num_scheduled_tokens=[4],
                    total_num_scheduled_tokens=4,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.dense_count, 0)
        self.assertEqual(plan.sparse_count, 4)
        self.assertTrue(plan.all_sparse)
        self.assertEqual(plan.dense_mask.numel(), 0)
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                self.assertTrue(
                    speclink_token_dense.current_verify_sparse_only_all_sparse(4)
                )
            finally:
                speclink_token_dense.end_verify_context(token)
        self.assertEqual(len(speclink_token_dense._pending_scores["req0"]), 1)

    def test_sparse_only_decode_does_not_request_draft_scores(self) -> None:
        with (
            mock.patch.object(speclink_token_dense, "enabled", return_value=True),
            mock.patch.object(
                speclink_token_dense,
                "linear_strategy",
                return_value="sparse_only_decode",
            ),
        ):
            self.assertFalse(speclink_token_dense.draft_scores_required())

        with (
            mock.patch.object(speclink_token_dense, "enabled", return_value=True),
            mock.patch.object(
                speclink_token_dense,
                "linear_strategy",
                return_value="full_sparse_residual",
            ),
        ):
            self.assertTrue(speclink_token_dense.draft_scores_required())

    def test_sparse_only_decode_keeps_prefill_rows_dense(self) -> None:
        with mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True):
            with mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "16",
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": "sparse_only_decode",
                },
            ):
                plan = speclink_token_dense.build_verify_dense_mask(
                    req_ids=["decode", "prefill"],
                    num_scheduled_tokens=[4, 3],
                    num_draft_tokens=[2, 0],
                    num_decode_draft_tokens=[2, -1],
                    cu_num_scheduled_tokens=[4, 7],
                    total_num_scheduled_tokens=7,
                    device=torch.device("cpu"),
                )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.dense_mask.tolist(),
            [False, False, False, False, True, True, True],
        )
        self.assertEqual(plan.dense_count, 3)
        self.assertEqual(plan.sparse_count, 4)
        self.assertTrue(plan.has_prefill_rows)
        self.assertEqual(plan.prefill_count, 3)
        self.assertEqual(plan.decode_count, 4)
        self.assertEqual(plan.prefill_rows[:3].tolist(), [4, 5, 6])
        self.assertEqual(plan.decode_rows[:4].tolist(), [0, 1, 2, 3])
        self.assertTrue(plan.contiguous_prefill_suffix)
        self.assertFalse(plan.contiguous_prefill_prefix)

    def test_sparse_transpose_policy_defaults_to_contiguous_outputs(self) -> None:
        class Module:
            def __init__(self, name: str) -> None:
                self._speclink_sparse24_module_name = name

        qkv = Module("model.layers.0.self_attn.qkv_proj")
        o_proj = Module("model.layers.0.self_attn.o_proj")
        down = Module("model.layers.0.mlp.down_proj")

        old = os.environ.pop("SPECLINK_SPARSE24_SKIP_TRANSPOSE", None)
        try:
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(qkv))
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(down))
        finally:
            if old is not None:
                os.environ["SPECLINK_SPARSE24_SKIP_TRANSPOSE"] = old

        with mock.patch.dict(
            os.environ, {"SPECLINK_SPARSE24_SKIP_TRANSPOSE": "layerwise"}
        ):
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(qkv))
            self.assertTrue(speclink_linear._skip_sparse_transpose_enabled(down))

        with mock.patch.dict(
            os.environ, {"SPECLINK_SPARSE24_SKIP_TRANSPOSE": "mlp"}
        ):
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(qkv))
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(o_proj))
            self.assertTrue(speclink_linear._skip_sparse_transpose_enabled(down))

        with mock.patch.dict(
            os.environ, {"SPECLINK_SPARSE24_SKIP_TRANSPOSE": "1"}
        ):
            self.assertTrue(speclink_linear._skip_sparse_transpose_enabled(qkv))
            self.assertTrue(speclink_linear._skip_sparse_transpose_enabled(down))

        with mock.patch.dict(
            os.environ, {"SPECLINK_SPARSE24_SKIP_TRANSPOSE": "0"}
        ):
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(qkv))
            self.assertFalse(speclink_linear._skip_sparse_transpose_enabled(down))

    def test_qkv_parallel_residual_is_limited_to_benchmarked_shape(self) -> None:
        with mock.patch.object(
            speclink_linear,
            "_QKV_PARALLEL_RESIDUAL_ENABLED",
            True,
        ):
            self.assertTrue(
                speclink_linear._is_qkv_parallel_residual_shape(
                    "model.layers.0.self_attn.qkv_proj",
                    4096,
                    6144,
                )
            )
            self.assertTrue(
                speclink_linear._is_qkv_parallel_residual_shape(
                    "model.layers.0.mlp.down_proj",
                    12288,
                    4096,
                )
            )
            self.assertFalse(
                speclink_linear._is_qkv_parallel_residual_shape(
                    "model.layers.0.self_attn.o_proj",
                    4096,
                    4096,
                )
            )
            self.assertFalse(
                speclink_linear._is_qkv_parallel_residual_shape(
                    "model.layers.0.self_attn.qkv_proj",
                    4096,
                    4096,
                )
            )

    def test_qkv_parallel_residual_requires_large_mixed_batch(self) -> None:
        module = SimpleNamespace(_speclink_qkv_parallel_residual=True)
        cuda_input = mock.Mock()
        cuda_input.shape = (288, 4096)
        cuda_input.is_cuda = True

        self.assertTrue(
            speclink_linear._should_parallel_qkv_residual(
                module,
                cuda_input,
                64,
            )
        )
        cuda_input.shape = (128, 4096)
        self.assertTrue(
            speclink_linear._should_parallel_qkv_residual(
                module,
                cuda_input,
                64,
            )
        )
        cuda_input.shape = (32, 4096)
        self.assertFalse(
            speclink_linear._should_parallel_qkv_residual(
                module,
                cuda_input,
                16,
            )
        )
        cuda_input.shape = (288, 4096)
        self.assertFalse(
            speclink_linear._should_parallel_qkv_residual(
                module,
                cuda_input,
                128,
            )
        )

    def test_residual_add_accepts_contiguous_full_output(self) -> None:
        full_out = torch.zeros((4, 3), dtype=torch.float16)
        row_add = torch.arange(6, dtype=torch.float16).reshape(3, 2).t()
        self.assertFalse(row_add.is_contiguous())
        self.assertEqual(tuple(row_add.stride()), (1, 2))
        row_indices = torch.tensor([1, 3], dtype=torch.int32)

        actual = speclink_linear._add_indexed_rows_(
            full_out,
            row_add,
            row_indices,
        )

        expected = torch.zeros((4, 3), dtype=torch.float16)
        expected[1] = row_add[0]
        expected[3] = row_add[1]
        self.assertIs(actual, full_out)
        self.assertTrue(torch.equal(actual, expected))

    def test_graph_eager_fallback_does_not_keep_shape_buffers(self) -> None:
        with (
            mock.patch.object(speclink_linear, "_REUSE_SPARSE_BUFFERS", True),
            mock.patch.object(speclink_linear, "_GRAPH_ROUTING_ENABLED", True),
            mock.patch.object(
                speclink_linear,
                "_CACHE_GRAPH_EAGER_FALLBACK",
                False,
            ),
            mock.patch.object(
                speclink_linear,
                "_cuda_graph_capturing",
                return_value=False,
            ),
        ):
            self.assertFalse(speclink_linear._reuse_sparse_buffers_enabled())

        with (
            mock.patch.object(speclink_linear, "_REUSE_SPARSE_BUFFERS", True),
            mock.patch.object(speclink_linear, "_GRAPH_ROUTING_ENABLED", True),
            mock.patch.object(
                speclink_linear,
                "_CACHE_GRAPH_EAGER_FALLBACK",
                False,
            ),
            mock.patch.object(
                speclink_linear,
                "_cuda_graph_capturing",
                return_value=True,
            ),
        ):
            self.assertTrue(speclink_linear._reuse_sparse_buffers_enabled())

        with (
            mock.patch.object(speclink_linear, "_REUSE_SPARSE_BUFFERS", True),
            mock.patch.object(speclink_linear, "_GRAPH_ROUTING_ENABLED", False),
            mock.patch.object(
                speclink_linear,
                "_cuda_graph_capturing",
                return_value=False,
            ),
        ):
            self.assertTrue(speclink_linear._reuse_sparse_buffers_enabled())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_residual_add_cuda_contiguous_fast_path(self) -> None:
        full_out = torch.zeros((8, 16), device="cuda", dtype=torch.float16)
        row_add = torch.randn((3, 16), device="cuda", dtype=torch.float16)
        row_indices = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int32)

        actual = speclink_linear._add_indexed_rows_(
            full_out,
            row_add,
            row_indices,
        )

        expected = torch.zeros_like(full_out)
        expected.index_add_(0, row_indices.long(), row_add)
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_copy_indexed_rows_accepts_rowmajor_output(self) -> None:
        full_out = torch.zeros((8, 16), device="cuda", dtype=torch.float16)
        row_values = torch.randn((3, 16), device="cuda", dtype=torch.float16)
        row_indices = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int32)

        sparse24_copy_indexed_rows_contiguous_(
            full_out,
            row_values,
            row_indices,
        )

        expected = torch.zeros_like(full_out)
        expected.index_copy_(0, row_indices.long(), row_values)
        self.assertTrue(torch.equal(full_out, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_split_dense_sparse_returns_contiguous_correct_output(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(31)
        rows, in_features, out_features = 48, 64, 32
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_matrix = torch.randn(
            (in_features, out_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        sparse_matrix, _ = apply_random_24_mask(
            dense_matrix,
            generator=generator,
        )
        packed = pack_24(sparse_matrix, layout="n_major")
        full_values, full_meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=in_features,
        )
        module = SimpleNamespace(
            weight=dense_matrix.t().contiguous(),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_selective_dense_enabled=True,
            _speclink_sparse24_linear_strategy="split_dense_sparse",
            _speclink_sparse24_full_a_values=full_values,
            _speclink_sparse24_full_a_meta_e=full_meta,
            _speclink_sparse24_module_name="test.linear",
        )
        sparse_rows = torch.arange(16, rows, device="cuda", dtype=torch.int32)
        dense_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
        dense_mask[sparse_rows.long()] = False
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=16,
            sparse_count=32,
            total_rows=rows,
        )

        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "split_dense_sparse"
                    )
                },
            ),
        ):
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                actual = speclink_linear.speclink_linear_forward(module, x)
            finally:
                speclink_token_dense.end_verify_context(token)
            torch.cuda.synchronize()

        expected = x @ dense_matrix
        expected[sparse_rows.long()] = x[sparse_rows.long()] @ sparse_matrix
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(
            torch.allclose(actual, expected, rtol=3e-2, atol=1.5e-1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_released_dense_weight_reconstructs_dense_output(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(37)
        rows, in_features, out_features = 48, 64, 32
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        original_weight = torch.randn(
            (out_features, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        keep, _method = speclink_structured_24._compute_keep_mask_24(
            original_weight,
            None,
        )
        module = SimpleNamespace(
            weight=torch.nn.Parameter(original_weight.clone(), requires_grad=False),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_24_mask_bytes=(
                speclink_structured_24._pack_keep_mask(keep)
            ),
            _speclink_24_row_scale=None,
        )
        stats: dict[str, object] = {}

        with mock.patch.dict(
            os.environ,
            {
                "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                    "full_sparse_residual"
                ),
                "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT": "1",
            },
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "test.linear",
                module.weight,
                stats,
            )
            self.assertEqual(module.weight.numel(), 0)
            self.assertTrue(module._speclink_sparse24_dense_weight_released)

            with mock.patch.object(
                speclink_token_dense,
                "_STATIC_ENABLED",
                True,
            ):
                actual = speclink_linear.speclink_linear_forward(module, x)
                torch.cuda.synchronize()

        expected = x @ original_weight.t()
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(
            torch.allclose(actual, expected, rtol=3e-2, atol=1.5e-1)
        )
        self.assertEqual(
            stats["speclink_kernel_released_dense_weight_module_names"],
            ["test.linear"],
        )

        retained_module = SimpleNamespace(
            weight=torch.nn.Parameter(original_weight.clone(), requires_grad=False),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_24_mask_bytes=speclink_structured_24._pack_keep_mask(keep),
            _speclink_24_row_scale=None,
        )
        retained_stats: dict[str, object] = {}
        with mock.patch.dict(
            os.environ,
            {
                "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                    "full_sparse_residual"
                ),
                "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT": "1",
                "SPECLINK_SPARSE24_RETAIN_DENSE_WEIGHT": "qkv",
            },
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                retained_module,
                "test.qkv_proj",
                retained_module.weight,
                retained_stats,
            )
        self.assertEqual(retained_module.weight.numel(), original_weight.numel())
        self.assertFalse(
            getattr(
                retained_module,
                "_speclink_sparse24_dense_weight_released",
                False,
            )
        )
        self.assertEqual(
            retained_stats["speclink_kernel_retained_dense_weight_module_names"],
            ["test.qkv_proj"],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qkv_cusparselt_residual_custom_op_matches_row_routing(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(101)
        rows, in_features, out_features = 576, 4096, 6144
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        original_weight = torch.randn(
            (out_features, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        keep, _method = speclink_structured_24._compute_keep_mask_24(
            original_weight,
            None,
        )
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                original_weight.clone(),
                requires_grad=False,
            ),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_24_mask_bytes=(
                speclink_structured_24._pack_keep_mask(keep)
            ),
            _speclink_24_row_scale=None,
        )
        stats: dict[str, object] = {}
        dense_rows = torch.arange(
            0,
            rows,
            9,
            device="cuda",
            dtype=torch.int64,
        )
        dense_mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
        dense_mask[dense_rows] = True
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=int(dense_rows.numel()),
            sparse_count=rows - int(dense_rows.numel()),
            total_rows=rows,
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_SPARSE24_QKV_CUSPARSELT": "1",
                },
            ),
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(
                speclink_linear,
                "_QKV_CUSPARSELT_ENABLED",
                True,
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.2.self_attn.qkv_proj",
                module.weight,
                stats,
            )
            token = speclink_token_dense.begin_verify_context(plan)
            try:
                actual = speclink_linear.speclink_linear_forward(module, x)
            finally:
                speclink_token_dense.end_verify_context(token)
            torch.cuda.synchronize()

            with (
                mock.patch.object(
                    speclink_token_dense,
                    "_STATIC_GRAPH_ROUTING",
                    True,
                ),
                mock.patch.object(
                    speclink_token_dense,
                    "_STATIC_NUM_SPEC_TOKENS",
                    8,
                ),
                mock.patch.dict(
                    os.environ,
                    {"SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0"},
                ),
            ):
                capture_plan = (
                    speclink_token_dense.pad_verify_plan_for_cudagraph(
                        plan,
                        actual_rows=rows,
                        padded_rows=rows,
                        device=torch.device("cuda"),
                    )
                )
                assert capture_plan is not None
                warmup_token = speclink_token_dense.begin_verify_context(
                    capture_plan
                )
                speclink_linear.speclink_linear_forward(module, x)
                speclink_token_dense.end_verify_context(warmup_token)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                capture_token = speclink_token_dense.begin_verify_context(
                    capture_plan
                )
                with torch.cuda.graph(graph):
                    graph_output = speclink_linear.speclink_linear_forward(
                        module,
                        x,
                    )
                speclink_token_dense.end_verify_context(capture_token)

                runtime_dense_rows = dense_rows + 1
                runtime_dense_mask = torch.zeros_like(dense_mask)
                runtime_dense_mask[runtime_dense_rows] = True
                runtime_plan = speclink_token_dense._make_plan(
                    runtime_dense_mask,
                    dense_count=int(runtime_dense_rows.numel()),
                    sparse_count=rows - int(runtime_dense_rows.numel()),
                    total_rows=rows,
                )
                speclink_token_dense.pad_verify_plan_for_cudagraph(
                    runtime_plan,
                    actual_rows=rows,
                    padded_rows=rows,
                    device=torch.device("cuda"),
                )
                graph.replay()
                torch.cuda.synchronize()

        sparse_weight = (
            original_weight.view(out_features, in_features // 4, 4)
            .masked_fill(~keep, 0)
            .view_as(original_weight)
        )
        expected = x @ sparse_weight.t()
        expected[dense_rows] = x[dense_rows] @ original_weight.t()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=4e-2, atol=2e-1),
            f"max_abs_diff={(actual.float() - expected.float()).abs().max().item()}",
        )
        graph_expected = x @ sparse_weight.t()
        graph_expected[runtime_dense_rows] = (
            x[runtime_dense_rows] @ original_weight.t()
        )
        self.assertTrue(
            torch.allclose(
                graph_output,
                graph_expected,
                rtol=4e-2,
                atol=2e-1,
            ),
            "CUDA graph replay did not consume the updated QKV row route",
        )
        self.assertTrue(
            isinstance(
                module._speclink_sparse24_qkv_cusparselt_packed,
                torch.Tensor,
            )
        )
        self.assertEqual(
            stats["speclink_kernel_qkv_cusparselt_module_names"],
            ["model.layers.2.self_attn.qkv_proj"],
        )

    def test_paired_inplace_o_dispatch_covers_bs16plus_active_waves(self) -> None:
        m128 = "128x32_full_128x32_residual_inplace"
        m256n32 = "256x32_full_256x32_residual_inplace"
        m256n64 = "256x64_full_256x32_residual_inplace"
        cases = {
            (112, 12): m128,
            (119, 32): m128,
            (126, 34): m256n32,
            (288, 32): m256n32,
            (294, 32): m256n64,
            (576, 32): m256n64,
            (583, 32): m256n32,
            (253, 64): m256n32,
            (259, 64): m256n64,
            (512, 64): m256n64,
            (513, 64): m256n32,
            (640, 64): m256n32,
            (649, 64): m256n64,
            (704, 64): m256n64,
        }
        with mock.patch.object(
            speclink_linear,
            "_PAIRED_INPLACE_O_ENABLED",
            True,
        ):
            for shape, expected in cases.items():
                with self.subTest(shape=shape):
                    self.assertEqual(
                        speclink_linear._paired_inplace_o_config(*shape),
                        expected,
                    )
            self.assertIsNone(speclink_linear._paired_inplace_o_config(111, 12))
            self.assertIsNone(speclink_linear._paired_inplace_o_config(705, 64))
            self.assertIsNone(speclink_linear._paired_inplace_o_config(144, 0))
            self.assertIsNone(speclink_linear._paired_inplace_o_config(144, 65))

    def test_paired_inplace_down_dispatch_covers_active_waves(self) -> None:
        m128 = "128x32_full_128x32_residual_inplace"
        m256n32 = "256x32_full_256x32_residual_inplace"
        m256n64 = "256x64_full_256x32_residual_inplace"
        cases = {
            (112, 12, 12288): m128,
            (119, 32, 14336): m128,
            (126, 34, 14336): m256n32,
            (288, 32, 12288): m256n32,
            (294, 32, 12288): m256n64,
            (576, 32, 12288): m256n64,
            (583, 32, 12288): m256n32,
            (640, 32, 12288): m256n32,
            (649, 32, 12288): m256n64,
            (253, 64, 14336): m256n32,
            (259, 64, 14336): m256n64,
            (512, 64, 14336): m256n64,
            (513, 64, 14336): m256n32,
            (640, 64, 14336): m256n32,
            (649, 64, 14336): m256n64,
        }
        with mock.patch.object(
            speclink_mlp,
            "_PAIRED_INPLACE_DOWN_ENABLED",
            True,
        ):
            for shape, expected in cases.items():
                with self.subTest(shape=shape):
                    self.assertEqual(
                        speclink_mlp._paired_inplace_down_config(*shape),
                        expected,
                    )
            self.assertIsNone(
                speclink_mlp._paired_inplace_down_config(111, 12, 12288)
            )
            self.assertIsNone(
                speclink_mlp._paired_inplace_down_config(705, 32, 12288)
            )
            self.assertIsNone(
                speclink_mlp._paired_inplace_down_config(144, 33, 12288)
            )
            self.assertIsNone(
                speclink_mlp._paired_inplace_down_config(144, 65, 14336)
            )
            self.assertIsNone(
                speclink_mlp._paired_inplace_down_config(144, 16, 8192)
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_paired_inplace_o_custom_op_tracks_cudagraph_route(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(107)
        rows, in_features, out_features = 144, 4096, 4096
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.1)
        original_weight = torch.randn(
            (out_features, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        keep, _method = speclink_structured_24._compute_keep_mask_24(
            original_weight,
            None,
        )
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                original_weight.clone(),
                requires_grad=False,
            ),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_24_mask_bytes=(
                speclink_structured_24._pack_keep_mask(keep)
            ),
            _speclink_24_row_scale=None,
        )
        stats: dict[str, object] = {}
        dense_rows = torch.arange(
            0,
            rows,
            9,
            device="cuda",
            dtype=torch.int64,
        )
        dense_mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
        dense_mask[dense_rows] = True
        plan = speclink_token_dense._make_plan(
            dense_mask,
            dense_count=int(dense_rows.numel()),
            sparse_count=rows - int(dense_rows.numel()),
            total_rows=rows,
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": "linear",
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "0",
                },
            ),
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(
                speclink_linear,
                "_PAIRED_INPLACE_O_ENABLED",
                True,
            ),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_GRAPH_ROUTING",
                True,
            ),
            mock.patch.object(
                speclink_token_dense,
                "_STATIC_NUM_SPEC_TOKENS",
                8,
            ),
        ):
            speclink_structured_24._attach_speclink_kernel_prepack(
                module,
                "model.layers.2.self_attn.o_proj",
                module.weight,
                stats,
            )
            capture_plan = speclink_token_dense.pad_verify_plan_for_cudagraph(
                plan,
                actual_rows=rows,
                padded_rows=rows,
                device=torch.device("cuda"),
            )
            assert capture_plan is not None
            warmup_token = speclink_token_dense.begin_verify_context(
                capture_plan
            )
            speclink_linear.speclink_linear_forward(module, x)
            speclink_token_dense.end_verify_context(warmup_token)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_token = speclink_token_dense.begin_verify_context(
                capture_plan
            )
            with torch.cuda.graph(graph):
                graph_output = speclink_linear.speclink_linear_forward(
                    module,
                    x,
                )
            speclink_token_dense.end_verify_context(capture_token)

            runtime_dense_rows = dense_rows + 1
            runtime_dense_mask = torch.zeros_like(dense_mask)
            runtime_dense_mask[runtime_dense_rows] = True
            runtime_plan = speclink_token_dense._make_plan(
                runtime_dense_mask,
                dense_count=int(runtime_dense_rows.numel()),
                sparse_count=rows - int(runtime_dense_rows.numel()),
                total_rows=rows,
            )
            speclink_token_dense.pad_verify_plan_for_cudagraph(
                runtime_plan,
                actual_rows=rows,
                padded_rows=rows,
                device=torch.device("cuda"),
            )
            graph.replay()
            torch.cuda.synchronize()

        sparse_weight = (
            original_weight.view(out_features, in_features // 4, 4)
            .masked_fill(~keep, 0)
            .view_as(original_weight)
        )
        expected = x @ sparse_weight.t()
        expected[runtime_dense_rows] = (
            x[runtime_dense_rows] @ original_weight.t()
        )
        self.assertTrue(
            torch.allclose(graph_output, expected, rtol=4e-2, atol=2e-1),
            "CUDA graph replay did not consume the updated O-projection route",
        )
        self.assertTrue(speclink_linear._PAIRED_INPLACE_O_COUNTERS)
        self.assertTrue(
            all(
                int(counters.abs().max().item()) == 0
                for counters in speclink_linear._PAIRED_INPLACE_O_COUNTERS.values()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dynamic_rows_update_captured_sparse_linear(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(29)
        rows, in_features, out_features = 48, 64, 32
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_matrix = torch.randn(
            (in_features, out_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        sparse_matrix, _ = apply_random_24_mask(
            dense_matrix,
            generator=generator,
        )

        def prepack(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            packed = pack_24(matrix, layout="n_major")
            return prepare_cutlass_sparse24_device_gemm(
                packed.values,
                packed.meta,
                layout=packed.layout,
                K=in_features,
            )

        full_values, full_meta = prepack(sparse_matrix)
        residual_values, residual_meta = prepack(dense_matrix - sparse_matrix)
        module = SimpleNamespace(
            weight=dense_matrix.t().contiguous(),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_selective_dense_enabled=True,
            _speclink_sparse24_linear_strategy="full_sparse_residual",
            _speclink_sparse24_full_a_values=full_values,
            _speclink_sparse24_full_a_meta_e=full_meta,
            _speclink_sparse24_residual_a_values=residual_values,
            _speclink_sparse24_residual_a_meta_e=residual_meta,
            _speclink_sparse24_module_name="test.linear",
        )

        def make_graph_plan(sparse_rows: list[int]):
            mask = torch.ones(rows, device="cuda", dtype=torch.bool)
            mask[torch.tensor(sparse_rows, device="cuda")] = False
            plan = speclink_token_dense._make_plan(
                mask,
                dense_count=rows - len(sparse_rows),
                sparse_count=len(sparse_rows),
                total_rows=rows,
            )
            return speclink_token_dense.pad_verify_plan_for_cudagraph(
                plan,
                actual_rows=rows,
                padded_rows=rows,
                device=torch.device("cuda"),
            )

        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "32",
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_residual"
                    ),
                },
            ),
        ):
            capture_plan = make_graph_plan(list(range(40, 48)))
            assert capture_plan is not None
            warmup_token = speclink_token_dense.begin_verify_context(capture_plan)
            speclink_linear.speclink_linear_forward(module, x)
            speclink_token_dense.end_verify_context(warmup_token)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_token = speclink_token_dense.begin_verify_context(capture_plan)
            with torch.cuda.graph(graph):
                graph_output = speclink_linear.speclink_linear_forward(module, x)
            speclink_token_dense.end_verify_context(capture_token)

            runtime_sparse_rows = list(range(8))
            runtime_plan = make_graph_plan(runtime_sparse_rows)
            assert runtime_plan is not None
            graph.replay()
            torch.cuda.synchronize()

        expected = x @ dense_matrix
        expected[runtime_sparse_rows] = (
            x[runtime_sparse_rows] @ sparse_matrix
        )
        self.assertTrue(
            torch.allclose(graph_output, expected, rtol=3e-2, atol=1.5e-1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dynamic_rows_update_captured_dense_override_linear(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(43)
        actual_rows, rows, in_features, out_features = 40, 48, 64, 32
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_matrix = torch.randn(
            (in_features, out_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        sparse_matrix, _ = apply_random_24_mask(
            dense_matrix,
            generator=generator,
        )
        packed = pack_24(sparse_matrix, layout="n_major")
        full_values, full_meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=in_features,
        )
        module = SimpleNamespace(
            weight=dense_matrix.t().contiguous(),
            bias=None,
            return_bias=False,
            input_is_parallel=True,
            _speclink_selective_dense_enabled=True,
            _speclink_sparse24_linear_strategy="full_sparse_dense_override",
            _speclink_sparse24_full_a_values=full_values,
            _speclink_sparse24_full_a_meta_e=full_meta,
            _speclink_sparse24_module_name="test.linear",
        )

        def make_graph_plan(dense_rows: list[int]):
            mask = torch.zeros(actual_rows, device="cuda", dtype=torch.bool)
            mask[torch.tensor(dense_rows, device="cuda")] = True
            plan = speclink_token_dense._make_plan(
                mask,
                dense_count=len(dense_rows),
                sparse_count=actual_rows - len(dense_rows),
                total_rows=actual_rows,
            )
            return speclink_token_dense.pad_verify_plan_for_cudagraph(
                plan,
                actual_rows=actual_rows,
                padded_rows=rows,
                device=torch.device("cuda"),
            )

        with (
            mock.patch.object(speclink_token_dense, "_STATIC_ENABLED", True),
            mock.patch.object(speclink_token_dense, "_STATIC_GRAPH_ROUTING", True),
            mock.patch.object(speclink_token_dense, "_STATIC_NUM_SPEC_TOKENS", 8),
            mock.patch.dict(
                os.environ,
                {
                    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": "16",
                    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": (
                        "full_sparse_dense_override"
                    ),
                },
            ),
        ):
            capture_dense_rows = list(range(24))
            capture_plan = make_graph_plan(capture_dense_rows)
            assert capture_plan is not None
            warmup_token = speclink_token_dense.begin_verify_context(capture_plan)
            speclink_linear.speclink_linear_forward(module, x)
            speclink_token_dense.end_verify_context(warmup_token)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_token = speclink_token_dense.begin_verify_context(capture_plan)
            with torch.cuda.graph(graph):
                graph_output = speclink_linear.speclink_linear_forward(module, x)
            speclink_token_dense.end_verify_context(capture_token)

            runtime_dense_rows = list(range(actual_rows - 24, actual_rows))
            runtime_plan = make_graph_plan(runtime_dense_rows)
            assert runtime_plan is not None
            graph.replay()
            torch.cuda.synchronize()

        expected = x @ sparse_matrix
        expected[runtime_dense_rows] = x[runtime_dense_rows] @ dense_matrix
        self.assertTrue(
            torch.allclose(graph_output, expected, rtol=3e-2, atol=1.5e-1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_exact_fused_kernels_match_row_policy(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(47)
        rows, in_features = 24, 64
        x = torch.randn(
            (rows, in_features),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_rows = torch.tensor(
            [0, 3, 7, 11, 14, 19, 23],
            device="cuda",
            dtype=torch.int32,
        )
        route_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
        route_mask[dense_rows.long()] = False
        sparse_rows = (
            route_mask.nonzero().flatten().to(torch.int32).contiguous()
        )
        dense_slot_by_row = torch.full(
            (rows,), -1, device="cuda", dtype=torch.int32
        )
        dense_slot_by_row[dense_rows.long()] = torch.arange(
            dense_rows.numel(), device="cuda", dtype=torch.int32
        )

        def prepare_exact(
            out_features: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            dense = torch.randn(
                (in_features, out_features),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            sparse, _ = apply_random_24_mask(dense, generator=generator)

            def prepack(
                matrix: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                packed = pack_24(matrix, layout="n_major")
                return prepare_cutlass_sparse24_device_gemm(
                    packed.values,
                    packed.meta,
                    layout=packed.layout,
                    K=in_features,
                )

            full_values, full_meta = prepack(sparse)
            residual_values, residual_meta = prepack(dense - sparse)
            return (
                dense,
                sparse,
                full_values,
                full_meta,
                residual_values,
                residual_meta,
            )

        linear = prepare_exact(128)
        linear_actual = sparse24_cutlass_routed_exact_linear_prepacked(
            x,
            linear[2],
            linear[3],
            linear[4],
            linear[5],
            dense_rows,
            sparse_rows,
            config="128x32x64_s4_sw4",
        )
        grouped_linear_actual = (
            sparse24_cutlass_grouped_owner_linear_prepacked(
                x,
                linear[2],
                linear[3],
                linear[4],
                linear[5],
                dense_rows,
                group_tiles=2,
                config="64x32x64_s3",
            )
        )

        gate = prepare_exact(256)
        gate_actual = sparse24_cutlass_routed_exact_swiglu_prepacked(
            x,
            gate[2],
            gate[3],
            gate[4],
            gate[5],
            dense_rows,
            sparse_rows,
            config="256x32x64_s3_sw4",
        )
        grouped_gate_actual = (
            sparse24_cutlass_grouped_owner_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                gate[4],
                gate[5],
                dense_rows,
                dense_slot_by_row,
                group_tiles=2,
                config="256x32x64_s3_sw4",
            )
        )
        approx_gate_actual, approx_dense_base = (
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slot_by_row,
                dense_count=int(dense_rows.numel()),
                config="256x32x64_s3_sw4",
                write_dense_approx=True,
            )
        )
        gate_residual = x[dense_rows.long()] @ (gate[0] - gate[1])
        gate_delta = torch.empty(
            (dense_rows.numel(), 128),
            device="cuda",
            dtype=torch.float16,
        )
        sparse24_routed_swiglu_delta_(
            approx_dense_base, gate_residual, gate_delta
        )
        dense_x = torch.zeros(
            (8, in_features), device="cuda", dtype=torch.float16
        )
        dense_x[: dense_rows.numel()] = x[dense_rows.long()]
        fused_gate_delta = torch.empty(
            (8, 128), device="cuda", dtype=torch.float16
        )
        sparse24_cutlass_residual_delta_swiglu_prepacked(
            dense_x,
            gate[4],
            gate[5],
            approx_dense_base,
            fused_gate_delta,
            config="256x32x64_s3_sw4",
        )
        torch.cuda.synchronize()

        linear_expected = x @ linear[1]
        linear_expected[dense_rows.long()] = (
            x[dense_rows.long()] @ linear[0]
        )
        self.assertTrue(
            torch.allclose(
                linear_actual,
                linear_expected,
                rtol=3e-2,
                atol=1.5e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                grouped_linear_actual,
                linear_expected,
                rtol=3e-2,
                atol=1.5e-1,
            )
        )
        gate_projected = x @ gate[1]
        gate_projected[dense_rows.long()] = x[dense_rows.long()] @ gate[0]
        gate_expected = torch.nn.functional.silu(
            gate_projected[:, :128]
        ) * gate_projected[:, 128:]
        sparse_gate_projected = x @ gate[1]
        approx_gate_expected = torch.nn.functional.silu(
            sparse_gate_projected[:, :128]
        ) * sparse_gate_projected[:, 128:]
        self.assertTrue(
            torch.allclose(
                gate_actual,
                gate_expected,
                rtol=4e-2,
                atol=2e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                grouped_gate_actual,
                gate_expected,
                rtol=4e-2,
                atol=2e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                approx_gate_actual,
                approx_gate_expected,
                rtol=4e-2,
                atol=2e-1,
            )
        )
        reconstructed_gate = approx_gate_actual.clone()
        reconstructed_gate[dense_rows.long()] += gate_delta
        self.assertTrue(
            torch.allclose(
                reconstructed_gate,
                gate_expected,
                rtol=5e-2,
                atol=3e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                fused_gate_delta[: dense_rows.numel()],
                gate_delta,
                rtol=5e-2,
                atol=3e-1,
            )
        )
        self.assertEqual(
            int(torch.count_nonzero(fused_gate_delta[dense_rows.numel() :])),
            0,
        )

    def test_transposed_sparse_input_fast_path_is_opt_in(self) -> None:
        old = os.environ.pop("SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT", None)
        try:
            self.assertFalse(speclink_linear._use_transposed_sparse_inputs_enabled())
        finally:
            if old is not None:
                os.environ["SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT"] = old

        with mock.patch.dict(
            os.environ, {"SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT": "1"}
        ):
            self.assertTrue(speclink_linear._use_transposed_sparse_inputs_enabled())


if __name__ == "__main__":
    unittest.main()
