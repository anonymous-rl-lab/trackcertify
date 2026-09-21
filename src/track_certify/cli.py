"""Command-line interface: ``track-certify demo | tstar``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _cmd_demo(args: argparse.Namespace) -> int:
    from . import Certifier
    from .instances import solvable_instance
    from . import _vendor  # noqa: F401
    from finite_ray_model import true_means  # noqa: E402

    inst = solvable_instance(args.r, args.c1, args.c2)
    print(inst.description)
    print(f"closed-form w* = ({inst.w_star_closed_form[0]:.3f}, "
          f"{inst.w_star_closed_form[1]:.3f})")
    rng = np.random.default_rng(args.seed)
    chol = np.linalg.cholesky(inst.model.sigma)
    means = true_means(inst.model, inst.truth, inst.amplitudes)
    cert = Certifier(inst.model, n_environments=inst.n_environments,
                     delta=args.delta, cap=args.cap)
    out = None
    while True:
        e = cert.next_environment()
        y = rng.normal(size=inst.model.dimension) @ chol.T + means[e]
        out = cert.observe(y)
        if out.stopped:
            break
    if out.refused:
        print(f"REFUSED at cap t={out.t} (no certificate issued)")
        return 1
    ok = out.decision == inst.truth
    print(f"stopped at t={out.t}, certificate={out.decision}, "
          f"counts={out.counts}")
    print(f"truth={inst.truth} -> {'CORRECT' if ok else 'WRONG'} "
          f"(guarantee: error probability <= {args.delta})")
    print(f"first-order budget T* log(1/delta) = "
          f"{inst.tstar_closed_form * np.log(1 / args.delta):.1f}")
    return 0


def _cmd_tstar(args: argparse.Namespace) -> int:
    from .api import build_model
    from .design import characteristic_time, coupling_tax

    spec = json.load(open(args.spec)) if args.spec else json.load(sys.stdin)
    sigma = np.asarray(spec["sigma"], dtype=float)
    orders = {k: tuple(v) for k, v in spec["graphs"].items()}
    hypothesis = (spec["hypothesis"]["graph"],
                  tuple(spec["hypothesis"]["targets"]))
    amplitudes = spec["amplitudes"]
    model = build_model(sigma, orders,
                        on_indistinct="warn" if args.allow_indistinct
                        else "raise")
    rep = characteristic_time(model, hypothesis, amplitudes)
    tax = coupling_tax(model, hypothesis, amplitudes)
    print(f"T*            = {rep.tstar:.6g}")
    print(f"w*            = {tuple(round(w, 4) for w in rep.w_star)}")
    print(f"alternatives  = {rep.n_alternatives}")
    print(f"active kinds  = {', '.join(rep.active_kinds)}")
    if tax.tax is not None:
        print(f"coupling tax  = {tax.tax:.4g}  "
              f"(joint {tax.tstar_joint:.4g} / "
              f"best conditional "
              f"{max(t for t in (tax.tstar_graph_only, tax.tstar_target_only) if t is not None):.4g})")
    if args.delta:
        print(f"budget at delta={args.delta}: "
              f"{rep.budget(args.delta):.1f} samples (first order)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="track-certify",
        description="Anytime joint certification of causal graphs and "
                    "unknown intervention targets")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run a certification demo on the exactly "
                                    "solvable coupled-binding instance")
    d.add_argument("--r", type=float, default=0.85)
    d.add_argument("--c1", type=float, default=1.0)
    d.add_argument("--c2", type=float, default=1.0)
    d.add_argument("--delta", type=float, default=0.01)
    d.add_argument("--cap", type=int, default=100_000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=_cmd_demo)

    t = sub.add_parser("tstar", help="characteristic time, optimal "
                                     "allocation, and coupling tax from a "
                                     "JSON spec")
    t.add_argument("spec", nargs="?", help="JSON file with keys sigma, "
                   "graphs, hypothesis{graph,targets}, amplitudes "
                   "(reads stdin if omitted)")
    t.add_argument("--delta", type=float, default=None)
    t.add_argument("--allow-indistinct", action="store_true")
    t.set_defaults(func=_cmd_tstar)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # clean CLI surface: no tracebacks for bad input
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
