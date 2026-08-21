# Week 1 — Ramulator 2.1 internals, and how to turn it into a 3D-stacked (Fabrik3D) memory simulator

This document is the Week-1 deliverable of the FabrikSim plan: a map of the
Ramulator 2.1 interfaces we will extend, and the concrete modification plan
for a logic-on-DRAM, Raptor-style 3D-stacked memory organization. Everything
here references actual files in this repo, and the end-to-end pipeline it
describes already runs (`fabrik/experiments/exp000_trace_smoke.py`).

---

## 1. How this Ramulator is put together

### 1.1 Component interfaces and the implementation registry

Every piece of the simulator is an implementation of one C++ interface,
registered by name so YAML/Python configs can select it at runtime:

| Interface | Where | Role | Stock implementations |
|---|---|---|---|
| `IFrontEnd` | `src/ramulator/frontend/i_frontend.h` | generates requests | `LoadStoreTrace`, `ReadWriteTrace`, `LatencyThroughputTrace`, `SimpleO3`, `BHO3`, `External` (pure-library mode) |
| `IMemorySystem` | `src/ramulator/memory_system/` | owns controllers, routes a request to a channel | `GenericDRAM` |
| `IChannelMapper` | picks the channel | `CacheLineInterleave`, `PassThroughChannelMapper`, `Gem5PortInterleave` |
| `IController` | `src/ramulator/controller/` | queues, schedules, issues DRAM commands | `GenericDDR`, `HBM12`, `HBM34`, `LPDDR5/6`, `GDDR7`, `PRAC`, `BlockHammer` |
| `IScheduler` | `controller/scheduler/` | picks next request | `FRFCFS`, ... |
| `IRowPolicy` | `controller/rowpolicy/` | open/closed page | `Open`, ... |
| `IRefreshManager` | `controller/refresh/impl/` | injects refresh | `AllBank`, `PerBank`, `HBM34PerBankRefresh`, `NoRefresh` |
| `IAddrMapper` | `controller/addr_mapper/impl/` | address bits → (channel, …, row, column) | `RoBaRaCoCh`, `ChRaBaRoCo`, `MOP4CLXOR`, `PassThroughAddrMapper` |
| `DRAMSpec` | `src/ramulator/dram/` | device organization, commands, timing | DDR3/4/5, LPDDR5/6, GDDR6/7, HBM1–4 |

Registration is one macro (`RAMULATOR_REGISTER_IMPLEMENTATION(IFace, Class,
"Name")`, or `..._DERIVED` for subclasses — see
`src/ramulator/controller/impl/hbm34_controller.cpp:9`). Adding a component
never touches existing code paths: new file + macro + CMake glob.

### 1.2 The Python DSL → codegen flow (this matters for us)

In this Ramulator 2.1 tree the DRAM standards are *authored in Python* and
compiled to C++:

```
python/ramulator/dram/hbm4.py        ← the human-edited spec (DSL)
        │  python -m ramulator codegen HBM4
        ▼
src/ramulator/dram/impl/HBM4.cpp     ← AUTO-GENERATED, do not edit
python/ramulator/dram/hbm4.py wrapper params  ← Python API stays in sync
```

A `DRAMStandard` subclass (`python/ramulator/dram/spec.py`) declares:

* `levels` — the hierarchy as an ordered dict of level name → initial state,
  e.g. HBM4: `Channel → PseudoChannel → Sid → BankGroup → Bank → Row → Column`;
* `commands`, `states`, `timing_params` — plain name lists;
* `timing_constraints` — `TimingConstraint(level, preceding, following,
  latency="nCL + nBL", window=, sibling=)` entries; the latency string is an
  expression over timing params, evaluated at config time;
* `org_presets` / `timing_presets` — named dicts of level counts and timing
  values, plus `resolve_secondary_timings()` for derived values (tRFC from
  density, tREFI from tCK, ...);
* `supported_requests` — request type → column command;
* `data_payload_bytes` / `internal_prefetch_size` / `channel_width` — the
  transaction size the memory system exposes:
  `get_tx_bytes() = data_payload_bytes` if set, else
  `prefetch × channel_width / 8` (`src/ramulator/dram/dram_spec.h:149`).

**Consequence: defining a brand-new memory organization is a Python-file
exercise**, not a C++ state-machine exercise. The generated C++ walks the
level tree (`dram/node.h`), applies the constraint table, and provides
ACT/PRE/RD/WR semantics from shared command templates
(`dram/commands/*.h`). This is the single biggest reason to fork this tree
for Fabrik rather than writing DRAM timing from scratch.

### 1.3 The simulation loop and clocking

`Simulation.run()` ticks frontend and memory system in a ratio set by the two
`clock_ratio` params. Inside the memory system, each controller ticks once
per memory-system tick; specs with faster data clocks use `tick_multiplier`
(HBM3/4 tick at half-CK granularity). A trace frontend finishes when every
line has been sent (`loadstore_trace.cpp: is_finished`); stats come back as a
nested dict (`sim.stats`) — per-controller row hits/misses/conflicts,
queue-length averages, read latency, achieved MB/s.

Two integration modes matter for FabrikSim:

* **Trace mode** (now): the analytical model emits `LD/ST` traces; Ramulator
  measures bandwidth/latency/row behavior. Zero C++ needed.
* **Library mode** (Week 4+): the outer compute simulator (TensorEngine,
  slices, NoC) calls `memory_system->send(req, callback)` through the
  `External` frontend / pure C++ library build, and the callback delivers
  `WEIGHT_FLIT_READY` events. This is the FabrikSim architecture from the
  plan — Ramulator stays the memory kernel, compute lives outside.

---

## 2. Modifying Ramulator for a 3D-stacked (logic-on-DRAM) memory

The target organization (Raptor-like, per chiplet):

```
Fabrik chiplet
 └── 16 slices
      └── slice: 16 TEs, TE_i ← channel_i ← 2/3/4 banks   (~256 ch/chiplet)
```

What is architecturally *different* from HBM, and where each difference
lands in the code:

### 2.1 New `Fabrik3D` DRAM spec — organization (Python DSL, no C++)

Create `python/ramulator/dram/fabrik3d.py`:

* **Levels:** `Channel → Bank → Row → Column`. No rank (single logical die
  stack per channel), no bank groups (no shared-IO grouping constraints), no
  pseudo-channels. 2/3/4-bank presets are just `"bank": 2|3|4` org presets.
* **Flit-wide vertical interface:** `data_payload_bytes = 128`. One column
  command moves one 128-B flit through µbumps — there is no pin-serialized
  burst, so `nBL = 1` and bandwidth is
  `channels × 128 B × f_column` rather than `pins × Gbps/pin`.
  The `rate`/`tCK_ps` timing param *is* the DRAM core clock
  (700 MHz → `tCK_ps = 1429`), not a doubled IO clock: `tick_multiplier = 1`.
* **Commands:** the minimal set — `ACT, PREpb, PREab, RD, WR, RDA, WRA,
  REFpb` (per-bank refresh only; see 2.4). All command templates already
  exist in `dram/commands/`.
* **Timing constraints:** keep the *bank-level* physics (`nRCD, nRAS, nRP,
  nRC, nRTP, nWR` — the DRAM array doesn't care that logic is bonded to it)
  and the column-cadence constraint (`RD→RD` at `nCCD` per channel, which
  sets the streaming rate). Drop the constraints that exist because of
  shared buses and power networks across many banks: `nFAW`, `nRRDS/L`
  across bank groups, cross-pseudo-channel rules. With 2–4 banks per
  channel and a private vertical bus, ACT-to-ACT across banks is limited
  only by power delivery — make it a single explicit `nRRD` param so it can
  be swept instead of inherited from JEDEC.
* **Secondary timings:** `resolve_secondary_timings()` computes tRFC/tREFI
  from per-bank density exactly as `hbm4.py` does — small banks means small
  tRFCpb, which is what makes per-channel refresh cheap.

Then `python -m ramulator codegen Fabrik3D` emits
`src/ramulator/dram/impl/Fabrik3D.cpp` and the Python wrapper params.

Validation of the org file alone: capacity per channel × 256 channels must
equal the chiplet capacity target; peak BW per channel must print as
`128 B / (nCCD × tCK)`.

### 2.2 `Fabrik3DController` — simpler than HBM, not fancier (small C++)

Derive from `ControllerBase` (as `generic_ddr_controller.cpp` does), *not*
from `HBMControllerBase` — the HBM base models the dual-edge shared CA bus
(row/column bus slots, rising/falling-edge pairing in
`hbm34_controller.cpp`), which does not exist when each channel has a
private through-silicon command path. The Fabrik controller is nearly the
generic controller with:

* short per-channel queues (one TE feeds one channel; queue depth is a
  research knob, start at 8–16);
* `Open` row policy + FRFCFS — with streaming traffic FRFCFS ≈ FCFS, and
  exp000 already shows 87% row hits / 98% of peak on pure streams, so the
  scheduler is not where Fabrik effort should go initially;
* **stream blocking** support: the one genuinely new mechanism. Expose a
  controller plugin (the `controller/plugin/` hook exists for exactly this)
  that recognizes consecutive-column requests and issues them as an
  uninterrupted column walk, reporting per-stream latency — this is what
  makes the simulator reproduce Raptor's ~2.5 ns streaming-flit behavior
  instead of quoting queueing-heavy random-access latency.

### 2.3 Address mapping — `FabrikAddressMapper` (tiny C++)

The Fabrik property is *locality by construction*: TE_i reads only
channel_i. Two-level approach:

* channel selection happens **above** Ramulator (the workload/trace already
  encodes which TE/channel a request belongs to) — use
  `PassThroughChannelMapper` semantics, or run one `GenericDRAM` per slice
  with 16 controllers and let high address bits pick the channel;
* within a channel, map `Row ← Column` LSB-first (`Ro-Ba-Co` with column
  bits lowest) so that a sequential expert-weight stream is a column walk
  within an open row, then a row increment, then (for 3-bank configs) a
  bank rotation that hides `nRP+nRCD` behind the other banks' streams.
  `ro_ba_ra_co_ch.cpp` is the 40-line template to copy.

The bank-rotation-on-row-crossing mapping is the mechanism by which a
3-bank channel streams at ~100% duty cycle; this is the first thing
experiment_001 must demonstrate.

### 2.4 Refresh — already per-channel, keep it that way

Refresh managers are instantiated **per controller** in this codebase, so
"one refreshing bank must not stall unrelated channels" is already true by
construction — a Fabrik channel refreshing affects only its own 2–4 banks.
Use `PerBank` (or a trivial `Fabrik3DPerBank` subclass) so that with ≥2
banks the idle bank refreshes while the other streams. What to *measure*
(not assume): the fraction of stream stalls attributable to refresh at
each bank count — that is the 2-vs-3-vs-4-bank tradeoff made quantitative.

### 2.5 Scale-out: slice → chiplet → card

Because channels share nothing below the NoC, simulate one slice
(16 channels) cycle-accurately and scale analytically until shared
resources appear — this is Phase 1/2/3 of the plan. Concretely:
`GenericDRAM(controllers=[16 × Fabrik3DController])` is a slice;
per-controller stats give `BW_slice` directly. The chiplet NoC and D2D live
in the outer FabrikSim, never inside Ramulator.

### 2.6 Energy (later, Week 5+)

Add an energy `controller_plugin` counting `ACT/PRE/RD/WR/REF` and bytes
moved, with pJ coefficients in the config: DRAM array pJ/access, vertical
I/O pJ/bit (calibrate to Raptor's measured ~0.4 pJ/bit), refresh energy.
Plugins see every issued command, so this needs no controller changes.

---

## 3. What exists already (this branch)

* `fabrik/analytical/` — the Days-1–3 analytical model: DeepSeek-V3/V2
  configs validated against published parameter counts (671 B/37 B active,
  236 B/21 B active), per-step weight/KV/FLOP accounting, expert-routing
  reuse statistics (closed form + Monte Carlo with Zipf skew), roofline
  sweeps, and a `LoadStoreTrace` generator whose expert streams are laid
  out contiguously per expert.
* `fabrik/experiments/exp000_trace_smoke.py` — end-to-end: analytical model
  → trace → cycle-accurate Ramulator (stock HBM3 channel). Result: 98.5% of
  channel peak bandwidth, 87% row-buffer hit rate — confirming the
  generated streams have the row-locality the Fabrik design depends on.
* `fabrik/tests/test_analytical.py` — 9 checks, all passing.

## 4. Week-2 task list (in implementation order)

1. `python/ramulator/dram/fabrik3d.py` (+ codegen) with 2/3/4-bank org
   presets and 500 MHz–1 GHz timing presets.
2. `FabrikAddressMapper` (`Ro-Ba-Co`, column LSBs, bank rotation).
3. `Fabrik3DController` = `GenericDDR` clone with short queues; stream
   plugin afterwards.
4. experiment_001: 16-channel slice frequency sweep (500 MHz–1 GHz),
   target: per-channel BW ≈ `128 B / (nCCD·tCK)` on expert streams, and a
   card-scaled curve passing through ~105 TB/s at 700 MHz for the
   Raptor-like configuration; wrong by >2× ⇒ the model is broken *before*
   anything Fabrik-specific is built on it.
