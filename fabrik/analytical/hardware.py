"""Roofline-style hardware model for decode.

Deliberately simple: one memory-bandwidth number and one effective compute
number per card. This is the layer the Ramulator-based FabrikMemory model
eventually replaces — the analytical model exists to (a) size traffic before
the simulator works and (b) sanity-check the simulator's outputs afterwards.

``step_time`` is reported under two composition rules:

* ``overlap`` — max(memory, compute): perfect overlap, the classic roofline;
* ``serial`` — memory + compute: no overlap, the pessimistic bound.

Real hardware lands between them; for B<=4 MoE decode the overlap bound is
usually the honest one because weight streaming and matvec pipeline cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decode import StepTraffic

TB = 1e12


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    mem_bw_bytes_per_s: float     # sustained DRAM read bandwidth per card
    compute_flops: float          # effective throughput at the chosen precision
    kv_bw_bytes_per_s: float | None = None  # separate KV path; None = shared

    @classmethod
    def simple(cls, name: str, mem_tb_s: float, compute_tflops: float) -> "HardwareConfig":
        return cls(name, mem_tb_s * TB, compute_tflops * 1e12)


# A Raptor-like card: ~105 TB/s measured stream bandwidth (dMatrix Raptor,
# ISCA) with a compute figure chosen so B=1 decode is memory-bound.
RAPTOR_LIKE = HardwareConfig.simple("raptor-like", 105.0, 500.0)


@dataclass(frozen=True)
class StepPerformance:
    hardware: str
    mem_time_s: float
    compute_time_s: float
    step_time_overlap_s: float
    step_time_serial_s: float
    tokens_per_s_overlap: float
    tokens_per_s_serial: float
    bound: str  # "memory" or "compute" under the overlap rule
    mem_bw_utilization_needed: float  # fraction of BW used if compute-bound


def evaluate(hw: HardwareConfig, t: StepTraffic) -> StepPerformance:
    if hw.kv_bw_bytes_per_s is not None:
        weight_time = (t.routed_weight_bytes + t.shared_weight_bytes) / hw.mem_bw_bytes_per_s
        kv_time = (t.kv_bytes + t.kv_write_bytes) / hw.kv_bw_bytes_per_s
        mem_time = max(weight_time, kv_time)
    else:
        mem_time = t.dram_bytes / hw.mem_bw_bytes_per_s
    compute_time = t.flops / hw.compute_flops
    overlap = max(mem_time, compute_time)
    serial = mem_time + compute_time
    return StepPerformance(
        hardware=hw.name,
        mem_time_s=mem_time,
        compute_time_s=compute_time,
        step_time_overlap_s=overlap,
        step_time_serial_s=serial,
        tokens_per_s_overlap=t.batch / overlap,
        tokens_per_s_serial=t.batch / serial,
        bound="memory" if mem_time >= compute_time else "compute",
        mem_bw_utilization_needed=mem_time / overlap,
    )
