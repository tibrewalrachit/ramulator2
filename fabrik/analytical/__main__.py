"""CLI for the MoE decode analytical model.

    python -m fabrik.analytical report --model deepseek-v3 --context 4096
    python -m fabrik.analytical trace  --model toy-moe --batch 4 -o step.trace
"""

from __future__ import annotations

import argparse
import sys

from .config import MODELS, PRECISIONS
from .report import (
    bandwidth_sweep,
    marginal_bandwidth_value,
    model_summary,
    traffic_table,
)
from .trace_gen import DecodeStepTracer, TraceConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="deepseek-v3", choices=sorted(MODELS))
    p.add_argument("--precision", default="fp8", choices=sorted(PRECISIONS))
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--zipf-alpha", type=float, default=0.0,
                   help="expert-popularity skew (0 = uniform routing)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fabrik.analytical")
    sub = ap.add_subparsers(dest="cmd")

    rp = sub.add_parser("report", help="traffic + roofline report (default)")
    _add_common(rp)
    rp.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
    rp.add_argument("--bw", type=float, nargs="+",
                    default=[52.5, 105.0, 150.0, 210.0, 300.0],
                    help="bandwidths to sweep, TB/s")
    rp.add_argument("--compute", type=float, default=500.0,
                    help="effective compute, TFLOPS at the chosen precision")

    tp = sub.add_parser("trace", help="emit a Ramulator2 LoadStoreTrace")
    _add_common(tp)
    tp.add_argument("--batch", type=int, default=1)
    tp.add_argument("-o", "--output", required=True)
    tp.add_argument("--flit-bytes", type=int, default=128)
    tp.add_argument("--max-layers", type=int, default=None)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--decimal-addr", action="store_true",
                    help="decimal addresses instead of 0x-hex")

    raw = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(raw)
    if args.cmd is None:
        args = ap.parse_args(["report"] + raw)

    cfg = MODELS[args.model]
    prec = PRECISIONS[args.precision]

    if args.cmd == "trace":
        tc = TraceConfig(flit_bytes=args.flit_bytes, max_layers=args.max_layers,
                         zipf_alpha=args.zipf_alpha, seed=args.seed,
                         hex_addr=not args.decimal_addr)
        tracer = DecodeStepTracer(cfg, prec, tc)
        with open(args.output, "w") as f:
            n = tracer.write(f, args.batch, args.context)
        print(f"wrote {n} requests ({n * args.flit_bytes / 1e9:.3f} GB of flits) "
              f"to {args.output}")
        return 0

    print(model_summary(cfg, prec))
    print()
    print(f"one decode step @ context {args.context}"
          + (f", zipf_alpha={args.zipf_alpha}" if args.zipf_alpha else
             " (uniform routing)"))
    table, traffics = traffic_table(cfg, prec, args.batch, args.context,
                                    zipf_alpha=args.zipf_alpha)
    print(table)
    print()
    print(f"roofline sweep (compute fixed at {args.compute:.0f} TFLOPS):")
    print(bandwidth_sweep(traffics, args.bw, args.compute))
    print()
    print("marginal value of bandwidth:")
    bws = sorted(args.bw)
    pairs = list(zip(bws, bws[1:]))
    print(marginal_bandwidth_value(traffics, pairs, args.compute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
