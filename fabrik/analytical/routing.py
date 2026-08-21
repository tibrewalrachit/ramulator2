"""Expert-routing statistics: how much weight reuse does a batch of B buy?

The quantity that drives routed-expert DRAM traffic is the number of *unique*
experts selected per MoE layer across the batch. Two estimators:

* :func:`expected_unique_experts` — closed form under uniform routing. With
  top-k selection each expert appears in one token's k-set with probability
  k/E, so across B independent tokens
  ``E[unique] = E * (1 - (1 - k/E)**B)``. Exact for uniform routing.

* :func:`simulate_unique_experts` — Monte Carlo with an optional Zipf-skewed
  expert popularity (real routers are not uniform; aux-loss-free balancing
  gets close, but skew is the pessimistic/realistic knob). Sampling is top-k
  without replacement via the Gumbel-top-k trick.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def expected_unique_experts(n_experts: int, top_k: int, batch: int) -> float:
    if batch <= 0:
        return 0.0
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


@dataclass(frozen=True)
class RoutingStats:
    mean_unique: float
    std_unique: float
    mean_max_share: float  # mean over trials of max tokens sharing one expert


def _zipf_weights(n: int, alpha: float) -> list[float]:
    if alpha <= 0.0:
        return [1.0] * n
    return [1.0 / (i + 1) ** alpha for i in range(n)]


def simulate_unique_experts(
    n_experts: int,
    top_k: int,
    batch: int,
    zipf_alpha: float = 0.0,
    trials: int = 2000,
    seed: int = 0,
) -> RoutingStats:
    rng = random.Random(seed)
    weights = _zipf_weights(n_experts, zipf_alpha)
    log_w = [math.log(w) for w in weights]

    uniques: list[int] = []
    max_shares: list[int] = []
    for _ in range(trials):
        counts: dict[int, int] = {}
        for _tok in range(batch):
            # Gumbel-top-k: top-k of log_w + Gumbel noise ~ weighted sampling
            # without replacement.
            keyed = [
                (log_w[e] - math.log(-math.log(rng.random())), e)
                for e in range(n_experts)
            ]
            keyed.sort(reverse=True)
            for _s, e in keyed[:top_k]:
                counts[e] = counts.get(e, 0) + 1
        uniques.append(len(counts))
        max_shares.append(max(counts.values()))

    n = len(uniques)
    mean_u = sum(uniques) / n
    var = sum((u - mean_u) ** 2 for u in uniques) / n
    return RoutingStats(
        mean_unique=mean_u,
        std_unique=math.sqrt(var),
        mean_max_share=sum(max_shares) / n,
    )


def sample_layer_experts(
    n_experts: int,
    top_k: int,
    batch: int,
    rng: random.Random,
    zipf_alpha: float = 0.0,
) -> list[list[int]]:
    """One routing draw: per token, the sorted list of selected expert ids.

    Used by the trace generator so a trace reflects a concrete routing
    outcome rather than an expectation.
    """
    log_w = [math.log(w) for w in _zipf_weights(n_experts, zipf_alpha)]
    out: list[list[int]] = []
    for _tok in range(batch):
        keyed = [
            (log_w[e] - math.log(-math.log(rng.random())), e)
            for e in range(n_experts)
        ]
        keyed.sort(reverse=True)
        out.append(sorted(e for _s, e in keyed[:top_k]))
    return out
