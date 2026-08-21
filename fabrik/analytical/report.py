"""Human-readable reports: the answers the analytical model exists to give."""

from __future__ import annotations

from .config import ModelConfig, Precision
from .decode import (
    StepTraffic,
    active_params_per_token,
    expert_params,
    step_traffic,
    total_params,
)
from .hardware import TB, HardwareConfig, evaluate
from .routing import expected_unique_experts, simulate_unique_experts

GB = 1e9


def _fmt_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:8.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:8.2f} MB"
    return f"{b / 1e3:8.2f} KB"


def model_summary(cfg: ModelConfig, prec: Precision) -> str:
    tot = total_params(cfg)
    act = active_params_per_token(cfg)
    lines = [
        f"model: {cfg.name}   precision: {prec.name} "
        f"(weights {prec.weight_bytes} B/elem, KV {prec.kv_bytes} B/elem)",
        f"  total params:      {tot / 1e9:9.1f} B   "
        f"({tot * prec.weight_bytes / 1e9:.0f} GB resident)",
        f"  active/token:      {act / 1e9:9.1f} B",
        f"  layers: {cfg.n_layers} ({cfg.n_dense_layers} dense + "
        f"{cfg.n_moe_layers} MoE), experts {cfg.n_routed_experts} routed "
        f"top-{cfg.n_active_experts} + {cfg.n_shared_experts} shared",
        f"  one routed expert: {expert_params(cfg) * prec.weight_bytes / 1e6:8.2f} MB "
        f"({expert_params(cfg) / 1e6:.1f} M params)",
        f"  KV cache/token:    {cfg.kv_cache_elems_per_token_per_layer} elems/layer "
        f"-> {cfg.n_layers * cfg.kv_cache_elems_per_token_per_layer * prec.kv_bytes / 1e3:.1f} KB/token",
    ]
    return "\n".join(lines)


def traffic_table(
    cfg: ModelConfig,
    prec: Precision,
    batches: list[int],
    context: int,
    zipf_alpha: float = 0.0,
    monte_carlo: bool = True,
) -> tuple[str, dict[int, StepTraffic]]:
    rows = []
    out: dict[int, StepTraffic] = {}
    header = (f"{'B':>3} {'uniq/layer':>10} {'routed W':>11} {'other W':>11} "
              f"{'KV read':>11} {'total DRAM':>11} {'/token':>11} "
              f"{'FLOP/tok':>9} {'AI':>7}")
    rows.append(header)
    rows.append("-" * len(header))
    for b in batches:
        if monte_carlo and zipf_alpha > 0.0:
            uniq = simulate_unique_experts(
                cfg.n_routed_experts, cfg.n_active_experts, b,
                zipf_alpha=zipf_alpha).mean_unique
        else:
            uniq = expected_unique_experts(
                cfg.n_routed_experts, cfg.n_active_experts, b)
        t = step_traffic(cfg, prec, b, context, uniq)
        out[b] = t
        rows.append(
            f"{b:>3} {uniq:>10.2f} {_fmt_bytes(t.routed_weight_bytes):>11} "
            f"{_fmt_bytes(t.shared_weight_bytes):>11} {_fmt_bytes(t.kv_bytes):>11} "
            f"{_fmt_bytes(t.dram_bytes):>11} {_fmt_bytes(t.dram_bytes / b):>11} "
            f"{t.flops / b / 1e9:>8.1f}G {t.arithmetic_intensity:>7.2f}")
    return "\n".join(rows), out


def bandwidth_sweep(
    traffics: dict[int, StepTraffic],
    bandwidths_tb_s: list[float],
    compute_tflops: float,
) -> str:
    header = (f"{'BW (TB/s)':>10} | "
              + " | ".join(f"B={b}: tok/s   bound" for b in traffics))
    rows = [header, "-" * len(header)]
    for bw in bandwidths_tb_s:
        hw = HardwareConfig("sweep", bw * TB, compute_tflops * 1e12)
        cells = []
        for b, t in traffics.items():
            p = evaluate(hw, t)
            cells.append(f"B={b}: {p.tokens_per_s_overlap:7.0f} {p.bound:>6}")
        rows.append(f"{bw:>10.0f} | " + " | ".join(cells))
    return "\n".join(rows)


def marginal_bandwidth_value(
    traffics: dict[int, StepTraffic],
    bw_pairs: list[tuple[float, float]],
    compute_tflops: float,
) -> str:
    """Answers 'is going from X to Y TB/s actually useful at this batch?'"""
    rows = []
    for lo, hi in bw_pairs:
        for b, t in traffics.items():
            p_lo = evaluate(HardwareConfig("lo", lo * TB, compute_tflops * 1e12), t)
            p_hi = evaluate(HardwareConfig("hi", hi * TB, compute_tflops * 1e12), t)
            gain = p_hi.tokens_per_s_overlap / p_lo.tokens_per_s_overlap - 1.0
            ideal = hi / lo - 1.0
            rows.append(
                f"  {lo:.0f} -> {hi:.0f} TB/s @ B={b}: +{gain * 100:5.1f}% tok/s "
                f"(ideal +{ideal * 100:.0f}%)"
                + ("  <- bandwidth fully used" if gain > 0.95 * ideal
                   else "  <- compute-limited" if gain < 0.05 * ideal else ""))
    return "\n".join(rows)
