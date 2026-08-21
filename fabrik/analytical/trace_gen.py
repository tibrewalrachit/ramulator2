"""Generate Ramulator 2 memory traces for one MoE decode step.

Output format matches Ramulator 2's ``LoadStoreTrace`` frontend
(``src/ramulator/frontend/impl/memory_trace/loadstore_trace.cpp``):
one request per line, ``LD <addr>`` / ``ST <addr>``, decimal or 0x-hex.

Address layout (all regions flit-aligned, laid out contiguously):

    [ per-layer weights: attn | router | shared experts | routed experts ]
    [ LM head ]
    [ per-sequence KV cache: layer-major, sequential within a layer ]

Expert weights are contiguous per expert, so streaming one expert produces
the long sequential column walks the Fabrik/Raptor design relies on; the
address mapper inside Ramulator decides how those spread over channels.

A full DeepSeek-V3 step is ~30 GB of reads (~10^8 flits) — generate traces
from ``toy-moe`` or a scaled config, or cap ``max_layers``, for tractable
simulations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, TextIO

from .config import ModelConfig, Precision
from .decode import (
    attn_params_per_layer,
    expert_params,
    ffn_params,
    embedding_params,
    router_params,
)
from .routing import sample_layer_experts


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass(frozen=True)
class TraceConfig:
    flit_bytes: int = 128
    max_layers: int | None = None   # cap layers for tractable trace sizes
    zipf_alpha: float = 0.0
    seed: int = 0
    hex_addr: bool = True
    kv_capacity_tokens: int = 8192  # allocated KV slots per sequence per layer


class DecodeStepTracer:
    """Lays out the model in a flat address space and emits one decode step."""

    def __init__(self, cfg: ModelConfig, prec: Precision, tc: TraceConfig):
        self.cfg, self.prec, self.tc = cfg, prec, tc
        f = tc.flit_bytes
        wb = prec.weight_bytes

        def wsize(elems: int) -> int:
            return _align_up(int(elems * wb), f)

        self.attn_size = wsize(attn_params_per_layer(cfg))
        self.dense_size = wsize(ffn_params(cfg.d_model, cfg.dense_intermediate))
        self.router_size = wsize(router_params(cfg))
        self.expert_size = wsize(expert_params(cfg))
        self.head_size = wsize(embedding_params(cfg))

        moe_block = (self.router_size
                     + cfg.n_shared_experts * self.expert_size
                     + cfg.n_routed_experts * self.expert_size)
        self.layer_sizes = [
            self.attn_size + (self.dense_size if l < cfg.n_dense_layers else moe_block)
            for l in range(cfg.n_layers)
        ]
        self.layer_base = []
        base = 0
        for s in self.layer_sizes:
            self.layer_base.append(base)
            base += s
        self.head_base = base
        self.kv_base = base + self.head_size

        kv_entry = _align_up(
            int(cfg.kv_cache_elems_per_token_per_layer * prec.kv_bytes), 1)
        self.kv_entry_bytes = kv_entry
        self.kv_layer_stride = _align_up(kv_entry * tc.kv_capacity_tokens, f)
        self.kv_seq_stride = self.kv_layer_stride * cfg.n_layers

    # -- region helpers ----------------------------------------------------

    def _stream(self, base: int, nbytes: int, op: str = "LD") -> Iterator[tuple[str, int]]:
        f = self.tc.flit_bytes
        for a in range(base, base + nbytes, f):
            yield op, a

    def _expert_base(self, layer: int, expert: int) -> int:
        return (self.layer_base[layer] + self.attn_size + self.router_size
                + (self.cfg.n_shared_experts + expert) * self.expert_size)

    def _kv_addr_range(self, seq: int, layer: int, n_tokens: int) -> tuple[int, int]:
        base = self.kv_base + seq * self.kv_seq_stride + layer * self.kv_layer_stride
        return base, self.kv_entry_bytes * n_tokens

    # -- the decode step ---------------------------------------------------

    def requests(self, batch: int, context: int) -> Iterator[tuple[str, int]]:
        """All (op, addr) requests for one decode step, in execution order."""
        cfg, tc = self.cfg, self.tc
        rng = random.Random(tc.seed)
        n_layers = cfg.n_layers if tc.max_layers is None else min(
            cfg.n_layers, tc.max_layers)
        f = tc.flit_bytes

        for layer in range(n_layers):
            # 1. attention weights, once per step
            yield from self._stream(self.layer_base[layer], self.attn_size)
            # 2. KV read walk per sequence, then append this step's entry
            for seq in range(batch):
                base, nbytes = self._kv_addr_range(seq, layer, context)
                yield from self._stream(base, _align_up(nbytes, f))
                yield "ST", (base + nbytes) // f * f  # append this step's entry
            if layer < cfg.n_dense_layers:
                # 3a. dense FFN
                yield from self._stream(
                    self.layer_base[layer] + self.attn_size, self.dense_size)
            else:
                # 3b. router + shared expert(s)
                base = self.layer_base[layer] + self.attn_size
                yield from self._stream(
                    base, self.router_size + cfg.n_shared_experts * self.expert_size)
                # 4. routed experts: union across the batch, each streamed once
                selected = sample_layer_experts(
                    cfg.n_routed_experts, cfg.n_active_experts, batch, rng,
                    tc.zipf_alpha)
                for e in sorted({e for tok in selected for e in tok}):
                    yield from self._stream(self._expert_base(layer, e),
                                            self.expert_size)
        if tc.max_layers is None or tc.max_layers >= cfg.n_layers:
            yield from self._stream(self.head_base, self.head_size)

    def write(self, out: TextIO, batch: int, context: int) -> int:
        n = 0
        fmt_hex = self.tc.hex_addr
        for op, addr in self.requests(batch, context):
            out.write(f"{op} 0x{addr:x}\n" if fmt_hex else f"{op} {addr}\n")
            n += 1
        return n
