"""Sanity checks for the MoE decode analytical model.

Run with ``pytest fabrik/tests`` or ``python fabrik/tests/test_analytical.py``.
"""

import io
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fabrik.analytical import (  # noqa: E402
    MODELS, PRECISIONS, DecodeStepTracer, TraceConfig,
    active_params_per_token, expected_unique_experts, evaluate,
    simulate_unique_experts, step_traffic, total_params, HardwareConfig,
)
from fabrik.analytical.hardware import TB  # noqa: E402


def test_deepseek_v3_param_counts():
    cfg = MODELS["deepseek-v3"]
    # Published: 671B total, 37B activated per token.
    assert abs(total_params(cfg) - 671e9) / 671e9 < 0.02
    assert abs(active_params_per_token(cfg) - 37e9) / 37e9 < 0.05


def test_deepseek_v2_param_counts():
    cfg = MODELS["deepseek-v2"]
    # Published: 236B total, 21B activated per token.
    assert abs(total_params(cfg) - 236e9) / 236e9 < 0.03
    assert abs(active_params_per_token(cfg) - 21e9) / 21e9 < 0.06


def test_mla_kv_cache_is_small():
    cfg = MODELS["deepseek-v3"]
    # MLA: 576 elems/token/layer, not heads * (qk + v) = 128 * 320.
    assert cfg.kv_cache_elems_per_token_per_layer == 576


def test_unique_experts_analytic_matches_monte_carlo():
    for b in (1, 2, 4, 8):
        analytic = expected_unique_experts(256, 8, b)
        mc = simulate_unique_experts(256, 8, b, trials=3000, seed=1)
        assert abs(mc.mean_unique - analytic) < 3 * (mc.std_unique / 50 + 0.1)


def test_unique_experts_bounds():
    assert expected_unique_experts(256, 8, 1) == 8.0
    u4 = expected_unique_experts(256, 8, 4)
    assert 8.0 < u4 < 32.0
    # Skewed routing must produce no more unique experts than uniform.
    skew = simulate_unique_experts(256, 8, 4, zipf_alpha=1.0, trials=2000)
    assert skew.mean_unique <= u4 + 0.5


def test_batch_amortizes_weight_traffic():
    cfg, prec = MODELS["deepseek-v3"], PRECISIONS["fp8"]
    per_tok = {}
    for b in (1, 4):
        u = expected_unique_experts(cfg.n_routed_experts, cfg.n_active_experts, b)
        t = step_traffic(cfg, prec, b, 4096, u)
        per_tok[b] = t.dram_bytes / b
    assert per_tok[4] < per_tok[1]  # weights amortize
    # ... but not 4x: routed experts barely overlap at B=4.
    assert per_tok[4] > per_tok[1] / 3.5


def test_roofline_memory_bound_at_b1():
    cfg, prec = MODELS["deepseek-v3"], PRECISIONS["fp8"]
    t = step_traffic(cfg, prec, 1, 4096,
                     expected_unique_experts(256, 8, 1))
    p = evaluate(HardwareConfig("hw", 105 * TB, 500e12), t)
    assert p.bound == "memory"
    # ~37GB/step at 105 TB/s -> a few thousand tok/s.
    assert 1000 < p.tokens_per_s_overlap < 10000


def test_trace_generation_parses_and_counts():
    cfg, prec = MODELS["toy-moe"], PRECISIONS["fp8"]
    tc = TraceConfig(flit_bytes=128, seed=7, kv_capacity_tokens=256)
    tracer = DecodeStepTracer(cfg, prec, tc)
    buf = io.StringIO()
    n = tracer.write(buf, batch=2, context=64)
    lines = buf.getvalue().splitlines()
    assert len(lines) == n > 0
    seen_st = False
    for line in lines:
        op, addr = line.split()
        assert op in ("LD", "ST")
        seen_st |= op == "ST"
        val = int(addr, 16)
        assert val % 128 == 0 and val >= 0
    assert seen_st  # KV appends present
    # Trace volume should be near the analytical weight+KV estimate
    # (loose bounds: the trace uses a concrete routing draw, not the mean).
    t = step_traffic(cfg, prec, 2, 64,
                     expected_unique_experts(cfg.n_routed_experts,
                                             cfg.n_active_experts, 2))
    flit_bytes = n * 128
    assert 0.5 * t.dram_bytes < flit_bytes < 2.0 * t.dram_bytes


def test_trace_deterministic_per_seed():
    cfg, prec = MODELS["toy-moe"], PRECISIONS["fp8"]
    outs = []
    for _ in range(2):
        buf = io.StringIO()
        DecodeStepTracer(cfg, prec, TraceConfig(seed=3)).write(buf, 2, 32)
        outs.append(buf.getvalue())
    assert outs[0] == outs[1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
