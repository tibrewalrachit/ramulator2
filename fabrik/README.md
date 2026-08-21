# FabrikSim — analytical MoE decode model + Ramulator bridge

Working directory for the Fabrik memory-subsystem study on top of
Ramulator 2.1. Current contents cover the first two steps of the plan:

* **Days 1–3** — `analytical/`: Python analytical model of DeepSeek-style
  MoE decode at batch 1–4 (weights/KV/FLOPs per token, expert-routing reuse,
  roofline, Ramulator trace generation).
* **Week 1** — `docs/week1_ramulator_3d_modification.md`: map of Ramulator's
  interfaces/controllers and the concrete plan for the `Fabrik3D`
  3D-stacked memory organization; `experiments/exp000_trace_smoke.py` runs
  the analytical model's trace through cycle-accurate Ramulator end to end.
* **Weeks 2–3** — the `Fabrik3D` organization itself, implemented in
  Ramulator (see "Fabrik3D in Ramulator" below), plus
  `experiments/exp001_fabrik_slice_sweep.py`: the frequency / bank-count
  calibration sweep.

## Quickstart

```sh
# traffic + roofline report for DeepSeek-V3 decode at B = 1/2/4
python3 -m fabrik.analytical report --model deepseek-v3 --context 4096

# same, with skewed (non-uniform) expert routing
python3 -m fabrik.analytical report --zipf-alpha 0.8

# emit a Ramulator2 LoadStoreTrace of one decode step (toy model)
python3 -m fabrik.analytical trace --model toy-moe --batch 4 -o step.trace

# end-to-end smoke test through cycle-accurate Ramulator (build first)
./build.sh
PYTHONPATH=python python3 fabrik/experiments/exp000_trace_smoke.py

# tests
python3 fabrik/tests/test_analytical.py     # or: pytest fabrik/tests
```

## What the analytical model says (DeepSeek-V3, fp8, context 4096)

Validated against published parameter counts (671 B total / 36.6 B active).

| B | unique experts/layer | DRAM/step | DRAM/token | bound at 105 TB/s + 500 TF |
|---|---|---|---|---|
| 1 | 8.0  | 36.8 GB | 36.8 GB | memory |
| 2 | 15.8 | 56.7 GB | 28.4 GB | ~balanced |
| 4 | 30.5 | 94.8 GB | 23.7 GB | compute |

Key structural facts the simulator work must respect:

* At B=1 there is essentially **no expert reuse** (8 unique of 256 per
  layer); batching to B=4 only amortizes per-token traffic by ~1.55×, not
  4×, because routed experts barely overlap. Routing skew *reduces* unique
  experts, i.e. uniform routing is the worst case for weight traffic.
* Weight streaming dominates: routed experts are ~20–78 GB/step vs
  ~0.14 GB/step of KV at context 4096 (MLA's 576-elem/token/layer cache).
  KV only matters at very long context.
* Whether more bandwidth helps is a pure roofline race with compute: at
  500 TFLOPS effective, B=1 stops scaling past ~130 TB/s. Sweep with
  `--bw`/`--compute` — this is the "is 200 TB/s useful at B=1?" question.

`exp000` result (toy model trace, one stock HBM3 pseudo-channel): 25.23 of
25.6 GB/s peak, 87% row-buffer hits — the generated expert streams have the
row locality the Fabrik3D design depends on.

## Fabrik3D in Ramulator

Three new components implement the 3D-stacked, logic-on-DRAM organization
(background in `docs/week1_ramulator_3d_modification.md`):

* `python/ramulator/dram/fabrik3d.py` — the `Fabrik3D` DRAM standard
  (codegen emits `src/ramulator/dram/impl/Fabrik3D.cpp`):
  `Channel -> Bank -> Row -> Column`, 1/2/3/4-bank org presets, one 128-B
  flit per column command (`data_payload_bytes = 128`, `nBL = 1`), `tCK` =
  the DRAM core clock with timing presets named `Fabrik3D_500MHz` ...
  `Fabrik3D_1000MHz`, bank-level array timings kept at conventional DRAM
  values, shared-bus constraints (tFAW, bank-group tRRD tiers) removed,
  per-bank refresh only.
* `src/ramulator/controller/addr_mapper/impl/fabrik_stream.cpp`
  (`FabrikStream`) — Row-Bank-Column mixed-radix mapping (3 banks is not a
  power of 2): sequential flits walk an open row's columns, row crossings
  rotate banks so the next ACT overlaps the current column walk.
* `src/ramulator/controller/scheduler/impl/fabrik_stream_scheduler.cpp`
  (`StreamAhead`) — the stream-blocking scheduler: issues timing-ready
  ACT/PRE to banks with no pending row-hits ahead of the stream. Without
  it, FRFCFS starves row commands behind always-ready row-hit reads and a
  streaming channel sits at ~73% of peak; with it, ~89% (the shared
  command slot's ceiling is 94.1%, per-bank refresh takes ~2.5%).

`exp001` results (200k-flit stream per channel, 3 banks): 79.7 GB/s per
channel at 700 MHz (89% of the 89.6 GB/s analytical peak), scaling
linearly 500 MHz -> 1 GHz; a ~105 TB/s Raptor-like card corresponds to
~1317 such channels (~5.1 chiplets of 256). The bank sweep quantifies the
design choice: 1 bank = 54% of peak (row turnaround exposed), 2+ banks =
89% (hidden); extra banks beyond 2 buy refresh margin, not bandwidth.

## Layout

```
fabrik/
├── analytical/          # the model: config / decode / routing / hardware /
│                        # report / trace_gen, CLI in __main__.py
├── experiments/         # numbered experiments (exp000 = trace smoke test)
├── docs/                # week1 Ramulator→Fabrik3D modification guide
└── tests/
```
