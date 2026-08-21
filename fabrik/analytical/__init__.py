"""Analytical model of large-MoE decode (B=1-4) for the Fabrik memory study.

Day-1 deliverable of the FabrikSim plan: per-token weight/KV/activation bytes
and FLOPs, expert-routing reuse statistics, a roofline performance estimate,
and a Ramulator 2 ``LoadStoreTrace`` generator for one decode step.
"""

from .config import MODELS, PRECISIONS, ModelConfig, Precision
from .decode import StepTraffic, active_params_per_token, step_traffic, total_params
from .hardware import RAPTOR_LIKE, HardwareConfig, StepPerformance, evaluate
from .routing import expected_unique_experts, simulate_unique_experts
from .trace_gen import DecodeStepTracer, TraceConfig

__all__ = [
    "MODELS", "PRECISIONS", "ModelConfig", "Precision",
    "StepTraffic", "active_params_per_token", "step_traffic", "total_params",
    "RAPTOR_LIKE", "HardwareConfig", "StepPerformance", "evaluate",
    "expected_unique_experts", "simulate_unique_experts",
    "DecodeStepTracer", "TraceConfig",
]
