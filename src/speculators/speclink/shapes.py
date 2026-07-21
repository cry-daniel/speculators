"""Canonical TP=1 fused linear shapes used by SpecLink experiments."""

from __future__ import annotations


# Checkpoint/vLLM weight orientation is [N, K].  At TP=1, vLLM fuses Q/K/V
# and gate/up along N for both model families.
TP1_FUSED_WEIGHT_SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    "qwen3_8b": {
        "qkv": (6144, 4096),
        "o": (4096, 4096),
        "gate_up": (24576, 4096),
        "down": (4096, 12288),
    },
    "llama3_1_8b": {
        "qkv": (6144, 4096),
        "o": (4096, 4096),
        "gate_up": (28672, 4096),
        "down": (4096, 14336),
    },
    "qwen3_14b": {
        # hidden=5120, heads=40, kv_heads=8, head_dim=128,
        # intermediate=17408.
        "qkv": (7168, 5120),
        "o": (5120, 5120),
        "gate_up": (34816, 5120),
        "down": (5120, 17408),
    },
    "qwen3_32b": {
        # Qwen3-32B uses a 128-wide head dimension independently of
        # hidden_size / num_attention_heads, so q_proj has N=8192.
        "qkv": (10240, 5120),
        "o": (5120, 8192),
        "gate_up": (51200, 5120),
        "down": (5120, 25600),
    },
    "llama3_70b": {
        # Llama 3/3.1 70B share these TP=1 linear shapes.
        "qkv": (10240, 8192),
        "o": (8192, 8192),
        "gate_up": (57344, 8192),
        "down": (8192, 28672),
    },
}
