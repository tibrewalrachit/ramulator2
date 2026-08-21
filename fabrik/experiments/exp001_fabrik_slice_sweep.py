"""experiment_001: Fabrik3D channel bandwidth — frequency and bank-count sweep.

Simulates one Fabrik3D channel (GenericDDR controller, StreamAhead
scheduler, FabrikStream Row-Bank-Column mapping, per-bank refresh)
streaming weight flits, cycle-accurately, against the analytical peak:

    BW_channel = 128 B / (nCCD * tCK)     (= 128 B x f at nCCD = 1)

Channels share nothing below the NoC — that is the defining Fabrik property
— so slice/chiplet/card bandwidth is channel bandwidth times channel count,
and one channel is the correct cycle-accurate unit. (Also a practical
matter: the LoadStoreTrace frontend issues at most one request per cycle
into the whole memory system, so a multi-channel sim under one trace
frontend measures the frontend, not the memory. exp001 v1 tripped exactly
this; the calibration gate caught it.)

This is the calibration gate from the plan: if the simulated channel does
not track the analytical curve (and, scaled to a Raptor-like card, pass
through ~105 TB/s at 700 MHz for a plausible channel count), the model is
wrong and nothing Fabrik-specific should be built on top of it.

Run from the repo root after ./build.sh:

    PYTHONPATH=python python3 fabrik/experiments/exp001_fabrik_slice_sweep.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import ramulator  # noqa: E402

FLIT = 128
N_FLITS = 200_000        # stream length; >> row size so ACT/refresh amortize
SLICE_CHANNELS = 16
CHIPLET_CHANNELS = 256
RAPTOR_CARD_TBS = 105.0  # measured Raptor-like card bandwidth target


def make_stream_trace(path: str) -> None:
    """A pure sequential weight stream — the Fabrik common case, and the
    same layout the analytical trace generator emits per expert."""
    with open(path, "w") as f:
        for i in range(N_FLITS):
            f.write(f"LD {i * FLIT}\n")


def run_channel(trace_path: str, freq_mhz: int, banks: int) -> dict:
    frontend = ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=trace_path)
    dram = ramulator.dram.Fabrik3D(
        org_preset=f"Fabrik3D_{banks}bank",
        timing_preset=f"Fabrik3D_{freq_mhz}MHz",
    )
    ctrl = ramulator.controller.GenericDDR(
        dram=dram,
        scheduler=ramulator.scheduler.StreamAhead(),
        refresh_manager=ramulator.refresh_manager.PerBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.FabrikStream(),
    )
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    sim = ramulator.Simulation(frontend, mem)
    sim.run()

    c = sim.stats["memory_system"]["controller"]
    served = c["num_read_reqs_served"]
    hits, misses, conflicts = (c["read_row_hits"], c["read_row_misses"],
                               c["read_row_conflicts"])
    lat = c["read_latency"] / max(served, 1)
    return {
        "bw_gbps": c["read_throughput_MBps"] / 1e3,
        "hit_pct": 100.0 * hits / max(hits + misses + conflicts, 1),
        "avg_lat_ns": lat * 1000.0 / freq_mhz,
    }


def main() -> None:
    scratch = os.environ.get("FABRIK_SCRATCH", "/tmp")
    trace = os.path.join(scratch, "exp001_stream.trace")
    make_stream_trace(trace)

    print(f"Fabrik3D channel: {N_FLITS} x {FLIT} B sequential flits, "
          f"FabrikStream Row-Bank-Column mapping, per-bank refresh\n")

    print("frequency sweep (3 banks/channel):")
    header = (f"{'MHz':>5} {'ch GB/s':>8} {'peak':>7} {'util':>6} "
              f"{'row hits':>9} {'lat ns':>7} {'slice':>9} {'chiplet':>10} "
              f"{'ch for 105TB/s':>15}")
    print(header)
    print("-" * len(header))
    results = {}
    for f in (500, 600, 700, 800, 900, 1000):
        r = run_channel(trace, f, banks=3)
        results[f] = r
        peak = FLIT * f * 1e6 / 1e9  # nCCD = 1
        bw = r["bw_gbps"]
        print(f"{f:>5} {bw:>8.2f} {peak:>7.1f} {bw / peak * 100:>5.1f}% "
              f"{r['hit_pct']:>8.1f}% {r['avg_lat_ns']:>7.1f} "
              f"{bw * SLICE_CHANNELS:>7.0f}GB "
              f"{bw * CHIPLET_CHANNELS / 1e3:>8.2f}TB "
              f"{RAPTOR_CARD_TBS * 1e3 / bw:>15.0f}")

    bw700 = results[700]["bw_gbps"]
    print(f"\nat 700 MHz: {bw700:.2f} GB/s/channel; a ~{RAPTOR_CARD_TBS:.0f} "
          f"TB/s Raptor-like card = {RAPTOR_CARD_TBS * 1e3 / bw700:.0f} "
          f"channels = {RAPTOR_CARD_TBS * 1e3 / bw700 / CHIPLET_CHANNELS:.1f} "
          f"chiplets of {CHIPLET_CHANNELS} channels")

    print("\nbank-count sweep (700 MHz): what hides row turnaround + refresh")
    peak700 = FLIT * 700 * 1e6 / 1e9
    for banks in (1, 2, 3, 4):
        r = run_channel(trace, 700, banks=banks)
        print(f"  {banks} bank{'s' if banks > 1 else ' '}: "
              f"{r['bw_gbps']:>6.2f} GB/s ({r['bw_gbps'] / peak700 * 100:5.1f}% "
              f"of peak), row hits {r['hit_pct']:5.1f}%, "
              f"lat {r['avg_lat_ns']:5.1f} ns")

    # Calibration gate. The hard ceiling is the shared command slot: one
    # command per cycle means each 32-flit row costs 32 RD + 1 ACT + 1 PRE
    # slots = 94.1% of peak; per-bank refresh (tRFCpb/nREFIpb, ~2.5%
    # effective) takes most of the rest. Streaming with >=2 banks must land
    # >=85% of the analytical peak at every frequency; the earlier FRFCFS
    # run sat at ~73% (row commands starved by ready row-hits) and a
    # 1-bank channel at ~54% (nothing hides row turnaround).
    cols_per_row = 32
    ceiling = cols_per_row / (cols_per_row + 2)
    print(f"\ncommand-slot ceiling: {ceiling * 100:.1f}% of peak "
          f"({cols_per_row} RD + ACT + PRE slots per row)")
    for f, r in results.items():
        peak = FLIT * f * 1e6 / 1e9
        assert r["bw_gbps"] / peak > 0.85, (
            f"{f} MHz: {r['bw_gbps']:.2f} GB/s < 85% of {peak:.1f} GB/s peak")
    print("calibration gate passed: >=85% of analytical peak at all "
          "frequencies (3-bank)")


if __name__ == "__main__":
    main()
