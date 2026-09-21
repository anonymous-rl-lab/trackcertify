"""Benchmark instances with theorem-known difficulty.

Two families from the accompanying paper in which membership, the
characteristic time, and the active alternatives are theorems rather than
linear-program outputs:

* :func:`solvable_instance` -- the exactly solvable coupled-binding family,
  built on the two-variable construction with which the paper exhibits an
  unbounded coupling tax (the family discussed under Proposition 1), plus an
  independent third node and a second environment: three nodes, two graphs,
  ``K = 2``, closed-form ``T* = 2/c2^2 + 2/(c1^2 (1 - r^2))``, the coupled
  alternative active for ``r > 1/sqrt(2)`` by that proposition's dual-mass
  equivalence, and ``T*`` divergent as ``r -> 1``.
* :func:`scalable_instance` -- the separable direct-sum family (supplement
  Corollary 1, multiple marginal tasks): ``m`` independent near-boundary
  pairs, ``2^m`` declared graphs, ``K = m``, additive
  ``T* = sum_i 2/(c_i^2 (1 - r_i^2))``, all ``m`` single-pair coupled
  alternatives active.

Both are ideal test problems: run any certification procedure on them and you
can compare its stopping behavior against a difficulty that is known exactly.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .api import build_model
from . import _vendor  # noqa: F401

from finite_ray_model import FiniteRayModel  # noqa: E402

Hypothesis = Tuple[str, Tuple[int, ...]]

_SQRT_HALF = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class Instance:
    """A ready-to-run benchmark instance.

    ``model`` and ``truth``/``amplitudes`` plug directly into
    :class:`track_certify.Certifier` (for streaming use) or
    :func:`track_certify.simulate` (for simulation studies);
    ``tstar_closed_form`` and ``w_star_closed_form`` are the theorem values.
    """

    model: FiniteRayModel
    truth: Hypothesis
    amplitudes: Tuple[float, ...]
    n_environments: int
    tstar_closed_form: float
    w_star_closed_form: Tuple[float, ...]
    description: str


def solvable_instance(r: float, c1: float = 1.0, c2: float = 1.0) -> Instance:
    """Exactly solvable coupled-binding instance.

    Parameters: correlation ``r`` in ``(1/sqrt(2), 1)`` (the coupled
    alternative binds exactly in this range) and nonzero intervention
    amplitudes ``c1`` (environment 1, targets node 0 of the correlated pair)
    and ``c2`` (environment 2, targets the independent node 2).
    """
    if not _SQRT_HALF < r < 1.0:
        raise ValueError("r must lie in (1/sqrt(2), 1) for the coupled "
                         "alternative to bind")
    if c1 == 0 or c2 == 0 or not np.isfinite([c1, c2]).all():
        raise ValueError("amplitudes must be nonzero and finite")
    sigma = np.array([[1.0, r, 0.0], [r, 1.0, 0.0], [0.0, 0.0, 1.0]])
    orders: Dict[str, Tuple[int, ...]] = {"G12": (0, 1, 2), "G21": (1, 0, 2)}
    model = build_model(sigma, orders)
    g1 = c1 * c1 * (1.0 - r * r) / 2.0
    g2 = c2 * c2 / 2.0
    tstar = 1.0 / g1 + 1.0 / g2
    w = np.array([1.0 / g1, 1.0 / g2])
    w = w / w.sum()
    return Instance(
        model=model,
        truth=("G12", (0, 2)),
        amplitudes=(float(c1), float(c2)),
        n_environments=2,
        tstar_closed_form=float(tstar),
        w_star_closed_form=(float(w[0]), float(w[1])),
        description=(f"solvable coupled-binding instance: r={r}, "
                     f"c=({c1},{c2}), T*={tstar:.4f}"),
    )


def scalable_instance(rs: Sequence[float], cs: Sequence[float],
                      singles: int = 0) -> Instance:
    """Separable direct-sum instance (supplement Corollary 1).

    ``m = len(rs)`` independent correlated pairs (each ``r_i`` in
    ``(1/sqrt(2), 1)``), ``singles`` isolated extra nodes, the ``2^m``
    orientation combinations declared as graphs, and environment ``i``
    targeting the first node of pair ``i`` with amplitude ``cs[i]``.
    Dimension ``d = 2m + singles``, ``K = m``, ``|C| = 2^m``.
    """
    rs = [float(r) for r in rs]
    cs = [float(c) for c in cs]
    if len(rs) != len(cs) or len(rs) < 2:
        raise ValueError("need m >= 2 pairs with one amplitude per pair")
    if any(not _SQRT_HALF < r < 1.0 for r in rs):
        raise ValueError("every r_i must lie in (1/sqrt(2), 1)")
    if any(c == 0 or not np.isfinite(c) for c in cs):
        raise ValueError("amplitudes must be nonzero and finite")
    if singles < 0:
        raise ValueError("singles must be nonnegative")
    m = len(rs)
    d = 2 * m + singles
    sigma = np.eye(d)
    for i, r in enumerate(rs):
        sigma[2 * i, 2 * i + 1] = sigma[2 * i + 1, 2 * i] = r
    orders: Dict[str, Tuple[int, ...]] = {}
    for bits in itertools.product((0, 1), repeat=m):
        seq: list = []
        for i, b in enumerate(bits):
            seq += [2 * i, 2 * i + 1] if b == 0 else [2 * i + 1, 2 * i]
        seq += list(range(2 * m, d))
        orders["G" + "".join(map(str, bits))] = tuple(seq)
    model = build_model(sigma, orders)
    g = [c * c * (1.0 - r * r) / 2.0 for r, c in zip(rs, cs)]
    tstar = float(sum(1.0 / gi for gi in g))
    w = np.array([1.0 / gi for gi in g])
    w = w / w.sum()
    return Instance(
        model=model,
        truth=("G" + "0" * m, tuple(2 * i for i in range(m))),
        amplitudes=tuple(cs),
        n_environments=m,
        tstar_closed_form=tstar,
        w_star_closed_form=tuple(float(x) for x in w),
        description=(f"scalable direct-sum instance: m={m}, d={d}, "
                     f"|C|={2 ** m}, T*={tstar:.4f}"),
    )
