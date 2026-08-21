"""Per-token / per-decode-step byte and FLOP accounting.

Everything is derived from :class:`~fabrik.analytical.config.ModelConfig` so a
preset can first be validated against the published parameter counts
(``total_params`` / ``active_params_per_token``) and only then used to
generate traffic numbers or Ramulator traces.

Traffic model for one decode step with batch B:

* attention, dense-FFN, shared-expert, router and LM-head weights are
  streamed once per step when ``weights_once_per_step`` (the natural
  weight-stationary design for B <= 4: a weight flit is applied to all B
  tokens while it sits in the tensor-engine buffer), or once per token
  otherwise;
* routed-expert weights are streamed once per *unique* expert selected across
  the batch in that layer (`routing.py` supplies the expected unique count);
* KV cache is read once per token per layer over the whole context — there is
  no cross-token reuse because the sequences are different;
* activations (residual stream in/out of every layer) are counted separately
  since they normally live in on-chip SRAM, not DRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, Precision


# ---------------------------------------------------------------------------
# Parameter counting (elements, not bytes)
# ---------------------------------------------------------------------------

def attn_params_per_layer(cfg: ModelConfig) -> int:
    d = cfg.d_model
    h = cfg.n_heads
    if cfg.kv_lora_rank is not None:
        q_in = cfg.q_lora_rank if cfg.q_lora_rank is not None else d
        p = 0
        if cfg.q_lora_rank is not None:
            p += d * cfg.q_lora_rank                       # W_DQ
        p += q_in * h * cfg.qk_head_dim                    # W_UQ
        p += d * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)  # W_DKV (+ rope key)
        p += cfg.kv_lora_rank * h * cfg.qk_nope_head_dim   # W_UK
        p += cfg.kv_lora_rank * h * cfg.v_head_dim         # W_UV
        p += h * cfg.v_head_dim * d                        # W_O
        return p
    n_kv = cfg.n_kv_heads if cfg.n_kv_heads is not None else h
    p = d * h * cfg.qk_head_dim                            # W_Q
    p += d * n_kv * (cfg.qk_head_dim + cfg.v_head_dim)     # W_K, W_V
    p += h * cfg.v_head_dim * d                            # W_O
    return p


def ffn_params(d_model: int, intermediate: int) -> int:
    # SwiGLU: gate, up, down
    return 3 * d_model * intermediate


def expert_params(cfg: ModelConfig) -> int:
    return ffn_params(cfg.d_model, cfg.moe_intermediate)


def router_params(cfg: ModelConfig) -> int:
    return cfg.d_model * cfg.n_routed_experts


def embedding_params(cfg: ModelConfig) -> int:
    return cfg.vocab_size * cfg.d_model


def total_params(cfg: ModelConfig) -> int:
    p = 2 * embedding_params(cfg)  # embedding + untied LM head
    p += cfg.n_layers * attn_params_per_layer(cfg)
    p += cfg.n_dense_layers * ffn_params(cfg.d_model, cfg.dense_intermediate)
    p += cfg.n_moe_layers * (
        (cfg.n_routed_experts + cfg.n_shared_experts) * expert_params(cfg)
        + router_params(cfg)
    )
    return p


def active_params_per_token(cfg: ModelConfig) -> int:
    """Parameters touched by one token (the "37B" number for DeepSeek-V3)."""
    p = embedding_params(cfg)  # LM head matvec; embedding lookup is one row
    p += cfg.n_layers * attn_params_per_layer(cfg)
    p += cfg.n_dense_layers * ffn_params(cfg.d_model, cfg.dense_intermediate)
    p += cfg.n_moe_layers * (
        (cfg.n_active_experts + cfg.n_shared_experts) * expert_params(cfg)
        + router_params(cfg)
    )
    return p


# ---------------------------------------------------------------------------
# Per-step traffic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepTraffic:
    """DRAM/SRAM traffic for one decode step of a batch of B tokens (bytes)."""

    batch: int
    context: int
    unique_experts_per_layer: float

    routed_weight_bytes: float     # routed experts, unique per layer per step
    shared_weight_bytes: float     # attn + dense FFN + shared experts + router + head
    kv_bytes: float                # KV-cache reads, all tokens, all layers
    kv_write_bytes: float          # new KV entries appended this step
    act_bytes: float               # residual stream traffic (usually SRAM)

    flops: float                   # total FLOPs for the step (all B tokens)

    @property
    def dram_read_bytes(self) -> float:
        return self.routed_weight_bytes + self.shared_weight_bytes + self.kv_bytes

    @property
    def dram_bytes(self) -> float:
        return self.dram_read_bytes + self.kv_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.dram_bytes


def flops_per_token(cfg: ModelConfig, context: int) -> float:
    """FLOPs for one token of decode at the given context length.

    2 FLOPs per weight element for every matmul against the streamed weights,
    plus attention score/value FLOPs over the context (in the MLA "absorbed"
    formulation the per-head score dim is kv_lora_rank + rope and the value
    dim is kv_lora_rank).
    """
    # active_params counts the LM head already; the embedding lookup is free.
    f = 2.0 * active_params_per_token(cfg)
    if cfg.kv_lora_rank is not None:
        score_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        value_dim = cfg.kv_lora_rank
        f += 2.0 * cfg.n_layers * cfg.n_heads * context * (score_dim + value_dim)
    else:
        f += 2.0 * cfg.n_layers * cfg.n_heads * context * (
            cfg.qk_head_dim + cfg.v_head_dim
        )
    return f


def step_traffic(
    cfg: ModelConfig,
    prec: Precision,
    batch: int,
    context: int,
    unique_experts_per_layer: float,
    weights_once_per_step: bool = True,
) -> StepTraffic:
    wb = prec.weight_bytes
    shared_stream = (
        cfg.n_layers * attn_params_per_layer(cfg)
        + cfg.n_dense_layers * ffn_params(cfg.d_model, cfg.dense_intermediate)
        + cfg.n_moe_layers * (cfg.n_shared_experts * expert_params(cfg)
                              + router_params(cfg))
        + embedding_params(cfg)  # LM head
    ) * wb
    if not weights_once_per_step:
        shared_stream *= batch

    routed = (
        cfg.n_moe_layers * unique_experts_per_layer * expert_params(cfg) * wb
    )

    kv_elems = cfg.kv_cache_elems_per_token_per_layer
    kv_read = batch * cfg.n_layers * context * kv_elems * prec.kv_bytes
    kv_write = batch * cfg.n_layers * kv_elems * prec.kv_bytes

    # Residual stream in+out of every layer, per token (SRAM-class traffic).
    act = batch * cfg.n_layers * 2 * cfg.d_model * prec.act_bytes

    return StepTraffic(
        batch=batch,
        context=context,
        unique_experts_per_layer=unique_experts_per_layer,
        routed_weight_bytes=routed,
        shared_weight_bytes=shared_stream,
        kv_bytes=kv_read,
        kv_write_bytes=kv_write,
        act_bytes=act,
        flops=batch * flops_per_token(cfg, context),
    )
