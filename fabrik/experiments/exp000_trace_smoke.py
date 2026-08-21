"""experiment_000: end-to-end smoke test of the analytical-model -> Ramulator path.

Generates one decode step of the toy MoE model as a LoadStoreTrace, runs it
through a stock HBM3 channel in cycle-accurate Ramulator 2, and reports the
achieved bandwidth and row-buffer behavior. This is the pipe that
experiment_001 (Raptor-like bandwidth reproduction) will reuse with the
Fabrik3D memory organization instead of HBM3.

Run from the repo root after ./build.sh:

    PYTHONPATH=python python3 fabrik/experiments/exp000_trace_smoke.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import ramulator  # noqa: E402

from fabrik.analytical import MODELS, PRECISIONS, DecodeStepTracer, TraceConfig  # noqa: E402


def generate_trace(path: str, batch: int, context: int, max_layers: int) -> int:
    cfg = MODELS["toy-moe"]
    prec = PRECISIONS["fp8"]
    tc = TraceConfig(flit_bytes=128, max_layers=max_layers, seed=0,
                     kv_capacity_tokens=1024, hex_addr=False)
    tracer = DecodeStepTracer(cfg, prec, tc)
    with open(path, "w") as f:
        return tracer.write(f, batch=batch, context=context)


def run(trace_path: str, n_reqs: int) -> None:
    frontend = ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=trace_path)
    hbm3 = ramulator.dram.HBM3(
        org_preset="HBM3_8Gb_8hi", timing_preset="HBM3_6400Mbps")
    ctrl = ramulator.controller.HBM34(
        dram=hbm3,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
    )
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    sim = ramulator.Simulation(frontend, mem)
    sim.run()

    ctrl_stats = sim.stats["memory_system"]["controller"]
    served = ctrl_stats["num_read_reqs_served"]
    hits = ctrl_stats["read_row_hits"]
    misses = ctrl_stats["read_row_misses"]
    conflicts = ctrl_stats["read_row_conflicts"]
    accesses = hits + misses + conflicts

    print(f"requests sent:      {n_reqs}")
    print(f"reads served:       {served}")
    print(f"read throughput:    {ctrl_stats['read_throughput_MBps'] / 1e3:.2f} GB/s "
          f"(one HBM3 6.4Gbps pseudo-channel peaks at 25.6 GB/s)")
    print(f"row-buffer hits:    {hits / accesses * 100:.1f}% "
          f"(misses {misses / accesses * 100:.1f}%, "
          f"conflicts {conflicts / accesses * 100:.1f}%)")
    print(f"avg read latency:   {ctrl_stats['read_latency'] / served:.1f} cycles")

    # The streaming layout should keep the row buffer hot; a shuffled trace
    # would sit near 0% hits. This is the property Fabrik3D depends on.
    assert hits / accesses > 0.7, "expected streaming-dominated row behavior"


def main() -> None:
    scratch = os.environ.get("FABRIK_SCRATCH", "/tmp")
    trace = os.path.join(scratch, "exp000_toy_step.trace")
    n = generate_trace(trace, batch=2, context=128, max_layers=2)
    print(f"trace: {trace} ({n} requests)")
    run(trace, n)


if __name__ == "__main__":
    main()
