"""Fabrik3D — a 3D-stacked, logic-on-DRAM memory organization.

Models a Raptor-style fine-grained channel: a small number of banks (2/3/4)
behind a private, flit-wide vertical interface bonded directly to a tensor
engine. Key departures from the JEDEC standards in this tree:

* ``Channel -> Bank -> Row -> Column`` — no ranks, bank groups, or
  pseudo-channels: nothing is shared between channels below the NoC.
* One column command moves one 128-B flit through the vertical bus
  (``data_payload_bytes = 128``, ``nBL = 1``): bandwidth is
  ``channels x 128 B x f_column``, never pins x Gbps/pin. ``tCK`` *is* the
  DRAM core clock, so timing presets are named by frequency
  (``Fabrik3D_700MHz``...), the primary sweep axis of experiment_001.
* Bank-level array physics (tRCD/tRAS/tRP/tRC/tWR/tRTP) are kept at
  conventional DRAM values — bonding logic to the array does not change the
  cell. Constraints that exist only because of shared buses and rails
  across many banks (tFAW, bank-group tRRD tiers, cross-pseudo-channel
  rules) are gone; ACT-to-ACT across the 2-4 banks is a single sweepable
  ``nRRD``.
* Per-bank refresh only (``REFpb``): a refreshing bank never stalls its
  channel's other banks, and never touches another channel (refresh
  managers are per-controller). Small banks -> small tRFCpb.

Column cadence ``nCCD`` defaults to 1 CK (the idealized wide-IO design:
one flit per core clock per channel). It is deliberately a plain timing
param so ``Fabrik3D(..., nCCD=2)`` sweeps slower array cadences.

Regenerate C++ after editing: ``python -m ramulator codegen Fabrik3D``.
"""

import math

from ramulator.dram.spec import DRAMStandard, TimingConstraint


class Fabrik3D(DRAMStandard):
    name = "Fabrik3D"
    internal_prefetch_size = 1  # no serialized burst: column address = flit
    tick_multiplier = 1         # tCK is the DRAM core clock
    data_payload_bytes = 128    # one flit per column command
    read_latency = "nCL + nBL"

    levels = {
        "Channel": "N_A",
        "Bank":    "Closed",
        "Row":     "Closed",
        "Column":  "N_A",
    }

    commands = ["ACT", "PREpb", "PREab", "RD", "WR", "RDA", "WRA", "REFpb"]

    states = ["Opened", "Closed", "N_A"]

    timing_params = [
        "rate", "nBL", "nCL", "nCWL",
        "nRCD", "nRP", "nRAS", "nRC", "nWR", "nRTP",
        "nCCD", "nRRD", "nRTW", "nWTR", "nPPD",
        "nRFCpb", "nREFIpb",
        "tCK_ps",
    ]

    supported_requests = {
        "Read": "RD",
        "Write": "WR",
    }

    timing_constraints = [
        # Channel level: the private vertical bus and shared channel logic.
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCD"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCD"),
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["WR", "WRA"], latency="nRTW"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nWTR"),
        TimingConstraint(level="Channel", preceding=["ACT"], following=["ACT"], latency="nRRD"),
        TimingConstraint(level="Channel", preceding=["PREpb", "PREab"], following=["PREpb", "PREab"], latency="nPPD"),
        # All-bank precharge interactions
        TimingConstraint(level="Channel", preceding=["ACT"], following=["PREab"], latency="nRAS"),
        TimingConstraint(level="Channel", preceding=["PREab"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="Channel", preceding=["RD"], following=["PREab"], latency="nRTP"),
        TimingConstraint(level="Channel", preceding=["WR"], following=["PREab"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Channel", preceding=["RDA"], following=["PREab"], latency="nRTP + nRP"),
        TimingConstraint(level="Channel", preceding=["WRA"], following=["PREab"], latency="nCWL + nBL + nWR + nRP"),

        # Bank level: DRAM array physics, unchanged by 3D stacking.
        TimingConstraint(level="Bank", preceding=["ACT"], following=["ACT"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["RD", "RDA"], latency="nRCD"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["WR", "WRA"], latency="nRCD"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["PREpb"], latency="nRAS"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RD"], following=["PREpb"], latency="nRTP"),
        TimingConstraint(level="Bank", preceding=["WR"], following=["PREpb"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Bank", preceding=["RDA"], following=["ACT", "REFpb"], latency="nRTP + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA"], following=["ACT", "REFpb"], latency="nCWL + nBL + nWR + nRP"),
        # Per-bank refresh: blocks only this bank, for only tRFCpb.
        TimingConstraint(level="Bank", preceding=["REFpb"], following=["ACT"], latency="nRFCpb"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["REFpb"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["REFpb"], latency="nRP"),
    ]

    # DRAM array physics in nanoseconds; converted per frequency preset.
    _ARRAY_NS = {
        "tCL": 14.0, "tCWL": 2.0,
        "tRCD": 14.0, "tRP": 14.0, "tRAS": 28.0,
        "tWR": 15.0, "tRTP": 7.5, "tRRD": 4.0, "tWTR": 5.0,
        "tPPD": 2.0,
        "tRFCpb": 100.0,   # small 3D banks refresh fast
    }
    _TREFI_NS = 3900.0     # per-bank refresh interval budget, JEDEC-like

    @classmethod
    def _make_timing_preset(cls, f_mhz):
        tck_ns = 1000.0 / f_mhz

        def n(ns):
            return max(1, math.ceil(ns / tck_ns))

        a = cls._ARRAY_NS
        p = {
            "rate": f_mhz,  # one flit slot per CK per channel
            "nBL": 1,
            "nCL": n(a["tCL"]), "nCWL": n(a["tCWL"]),
            "nRCD": n(a["tRCD"]), "nRP": n(a["tRP"]), "nRAS": n(a["tRAS"]),
            "nRC": n(a["tRAS"] + a["tRP"]),
            "nWR": n(a["tWR"]), "nRTP": n(a["tRTP"]),
            "nCCD": 1,
            "nRRD": n(a["tRRD"]), "nWTR": n(a["tWTR"]), "nPPD": n(a["tPPD"]),
            "nRFCpb": n(a["tRFCpb"]),
            "nREFIpb": -1,  # resolved from bank count in resolve_secondary_timings
            "tCK_ps": round(1_000_000 / f_mhz),
        }
        p["nRTW"] = max(1, p["nCL"] + p["nBL"] - p["nCWL"] + 1)
        return p

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        # Round-robin REFpb: each bank refreshed once per tREFI window.
        tCK_ps = timing_dict["tCK_ps"]
        n_banks = org_dict["bank"]
        timing_dict["nREFIpb"] = math.ceil(
            cls._TREFI_NS * 1000.0 / n_banks / tCK_ps)


def _org_preset(n_banks, rows=1 << 14, columns=32):
    # One channel: n_banks x rows x (columns x 128 B) rows.
    density_mbit = n_banks * rows * columns * 128 * 8 // (1 << 20)
    return {
        "density": density_mbit,
        "dq": 1024,             # 128-B flit-wide vertical interface
        "channel_width": 1024,
        "bank": n_banks,
        "row": rows,
        "column": columns,      # 32 x 128 B = 4 KB row
    }


Fabrik3D.org_presets = {
    "Fabrik3D_1bank": _org_preset(1),  # baseline: nothing hides row turnaround
    "Fabrik3D_2bank": _org_preset(2),
    "Fabrik3D_3bank": _org_preset(3),
    "Fabrik3D_4bank": _org_preset(4),
}

Fabrik3D.timing_presets = {
    f"Fabrik3D_{f}MHz": Fabrik3D._make_timing_preset(f)
    for f in (500, 600, 700, 800, 900, 1000)
}
