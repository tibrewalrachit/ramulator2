"""Model and numeric-precision configuration for the MoE decode analytical model.

All architecture parameters are expressed the way DeepSeek-family configs
express them (MLA low-rank dims, first-k-dense-replace, routed/shared experts)
so a preset can be checked against the published total / activated parameter
counts before any performance number is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Precision:
    """Bytes per element for each traffic class.

    Fractional values are allowed (e.g. 0.5 for INT4 weights).
    """

    name: str
    weight_bytes: float
    kv_bytes: float
    act_bytes: float


FP16 = Precision("fp16", 2.0, 2.0, 2.0)
FP8 = Precision("fp8", 1.0, 1.0, 2.0)
INT4_W = Precision("int4-weights", 0.5, 1.0, 2.0)  # W4A8-style: fp8 KV, fp16 acts

PRECISIONS = {p.name: p for p in (FP16, FP8, INT4_W)}


@dataclass(frozen=True)
class ModelConfig:
    """A decoder-only transformer with (optionally) MLA attention and MoE FFNs.

    ``q_lora_rank``/``kv_lora_rank`` of ``None`` means standard (non-MLA)
    projections; ``n_kv_heads`` then controls GQA. The first
    ``n_dense_layers`` layers use a dense FFN of width ``dense_intermediate``;
    the remaining layers are MoE with ``n_routed_experts`` routed experts of
    width ``moe_intermediate`` plus ``n_shared_experts`` always-on experts.
    """

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_dense_layers: int
    dense_intermediate: int
    moe_intermediate: int
    n_routed_experts: int
    n_active_experts: int
    n_shared_experts: int
    n_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    n_kv_heads: int | None = None  # only used when kv_lora_rank is None

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_elems_per_token_per_layer(self) -> int:
        """Elements cached per token per layer.

        MLA caches only the compressed KV latent plus the shared RoPE key;
        standard attention caches full K and V for each KV head.
        """
        if self.kv_lora_rank is not None:
            return self.kv_lora_rank + self.qk_rope_head_dim
        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        return n_kv * (self.qk_head_dim + self.v_head_dim)

    def scaled(self, **overrides) -> "ModelConfig":
        """A copy with some fields overridden (for what-if sweeps)."""
        return replace(self, **overrides)


DEEPSEEK_V3 = ModelConfig(
    name="deepseek-v3",
    vocab_size=129280,
    d_model=7168,
    n_layers=61,
    n_dense_layers=3,
    dense_intermediate=18432,
    moe_intermediate=2048,
    n_routed_experts=256,
    n_active_experts=8,
    n_shared_experts=1,
    n_heads=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    q_lora_rank=1536,
    kv_lora_rank=512,
)

DEEPSEEK_V2 = ModelConfig(
    name="deepseek-v2",
    vocab_size=102400,
    d_model=5120,
    n_layers=60,
    n_dense_layers=1,
    dense_intermediate=12288,
    moe_intermediate=1536,
    n_routed_experts=160,
    n_active_experts=6,
    n_shared_experts=2,
    n_heads=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    q_lora_rank=1536,
    kv_lora_rank=512,
)

# Small config for fast tests and short traces; not a real model.
TOY_MOE = ModelConfig(
    name="toy-moe",
    vocab_size=32000,
    d_model=1024,
    n_layers=4,
    n_dense_layers=1,
    dense_intermediate=4096,
    moe_intermediate=512,
    n_routed_experts=32,
    n_active_experts=4,
    n_shared_experts=1,
    n_heads=16,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    q_lora_rank=512,
    kv_lora_rank=128,
)

MODELS = {m.name: m for m in (DEEPSEEK_V3, DEEPSEEK_V2, TOY_MOE)}
