"""Experiment-design tools: characteristic time, optimal allocation, coupling tax.

These answer the *before-you-sample* questions: how hard is this joint
certification instance (its characteristic time ``T*``), how should a sampling
budget be split across environments (the max--min optimal allocation ``w*``),
which wrong answers dominate the cost (the active alternatives and their
kinds), and how much harder the joint problem is than its two conditional
relaxations (the coupling tax).

All quantities are computed by the same linear-program and divergence
code that the certifier itself uses (``_vendor/finite_ray_model.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import _vendor  # noqa: F401

from finite_ray_model import (  # noqa: E402
    FiniteRayModel,
    alternative_divergences,
    characteristic_time as _lp_characteristic_time,
)

Hypothesis = Tuple[str, Tuple[int, ...]]

_ACTIVE_TOL = 1e-9


@dataclass(frozen=True)
class DesignReport:
    """Output of :func:`characteristic_time`.

    Attributes
    ----------
    tstar:
        The characteristic time ``T*(theta)``: first-order samples per unit
        ``log(1/delta)`` that any delta-correct procedure needs, and that
        Track-and-Certify attains.
    w_star:
        Max--min optimal allocation over the K environments.
    game_value:
        ``1/T*`` -- the value of the profiled information game at ``w_star``.
    n_alternatives:
        Number of answer-changing alternatives in the declared class.
    active:
        The alternatives attaining the max--min value at ``w_star``
        (as ``(kind, hypothesis)`` pairs); these dominate the sample cost.
    active_kinds:
        Sorted set of kinds among the active alternatives; a ``"coupled"``
        entry means the nearest wrong answers change graph and targets
        together.
    """

    tstar: float
    w_star: Tuple[float, ...]
    game_value: float
    n_alternatives: int
    active: Tuple[Tuple[str, Hypothesis], ...]
    active_kinds: Tuple[str, ...]

    def budget(self, delta: float) -> float:
        """First-order sample estimate ``T* log(1/delta)`` (no transients)."""
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        return self.tstar * float(np.log(1.0 / delta))


@dataclass(frozen=True)
class CouplingReport:
    """Output of :func:`coupling_tax`.

    ``tax = tstar_joint / max(tstar_graph_only, tstar_target_only)``: how much
    the joint problem exceeds its harder conditional relaxation.  A tax near 1
    means a staged pipeline loses little *in the limit*; a large tax means the
    nearest wrong answer changes graph and targets together and any
    estimate-targets-then-certify-graph shortcut is first-order suboptimal.
    """

    tstar_joint: float
    tstar_graph_only: Optional[float]
    tstar_target_only: Optional[float]
    tstar_coupled_only: Optional[float]
    tax: Optional[float]


def _divergences(model: FiniteRayModel, hypothesis: Hypothesis,
                 amplitudes: Sequence[float]):
    amps = [float(a) for a in amplitudes]
    if len(amps) != len(hypothesis[1]):
        raise ValueError("need one amplitude per environment")
    if any(a == 0 or not np.isfinite(a) for a in amps):
        raise ValueError("amplitudes must be nonzero and finite")
    if hypothesis[0] not in model.graph_labels:
        raise ValueError(f"unknown graph label {hypothesis[0]!r}")
    rows, kinds, hyps = alternative_divergences(model, hypothesis, amps)
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        raise ValueError("the declared class contains no alternative to the "
                         "given hypothesis")
    return rows, list(kinds), list(hyps)


def characteristic_time(model: FiniteRayModel, hypothesis: Hypothesis,
                        amplitudes: Sequence[float]) -> DesignReport:
    """Characteristic time, optimal allocation, and active alternatives.

    Raises ``ValueError`` if the instance is unidentifiable within its class
    (some alternative has zero profiled divergence in every environment, so
    ``T*`` is infinite and no certificate can ever separate the pair).
    """
    rows, kinds, hyps = _divergences(model, hypothesis, amplitudes)
    if np.any(rows.max(axis=1) <= 0):
        raise ValueError(
            "instance is unidentifiable within its declared class: an "
            "alternative is indistinguishable from the hypothesis in every "
            "environment (gamma_J = 0, T* = infinity)")
    tstar, w = _lp_characteristic_time(rows)
    tstar = float(tstar)
    if not np.isfinite(tstar) or tstar <= 0 or w is None:
        raise ValueError("characteristic-time program did not return a "
                         "finite positive value")
    w = np.asarray(w, dtype=float)
    achieved = rows @ w
    active_idx = np.flatnonzero(achieved <= achieved.min() + _ACTIVE_TOL)
    active = tuple((kinds[i], hyps[i]) for i in active_idx)
    return DesignReport(
        tstar=tstar,
        w_star=tuple(float(x) for x in w),
        game_value=1.0 / tstar,
        n_alternatives=int(rows.shape[0]),
        active=active,
        active_kinds=tuple(sorted({k for k, _ in active})),
    )


def _restricted_tstar(rows: np.ndarray, kinds: Sequence[str],
                      keep: str) -> Optional[float]:
    mask = [i for i, k in enumerate(kinds) if k == keep]
    if not mask:
        return None
    sub = rows[mask]
    if np.any(sub.max(axis=1) <= 0):
        return float("inf")
    tstar, w = _lp_characteristic_time(sub)
    if w is None or not np.isfinite(tstar) or tstar <= 0:
        return None
    return float(tstar)


def coupling_tax(model: FiniteRayModel, hypothesis: Hypothesis,
                 amplitudes: Sequence[float]) -> CouplingReport:
    """Joint characteristic time against its conditional relaxations.

    The conditional times restrict the alternative set to one kind:
    graph-only (same targets, different graph), target-only (same graph,
    different targets), and coupled (both change).  The tax divides the joint
    ``T*`` by the larger of the two conditional times, matching the paper's
    definition; entries are ``None`` when the class contains no alternative of
    that kind.
    """
    rows, kinds, _ = _divergences(model, hypothesis, amplitudes)
    joint = characteristic_time(model, hypothesis, amplitudes).tstar
    tg = _restricted_tstar(rows, kinds, "graph")
    tt = _restricted_tstar(rows, kinds, "target")
    tc = _restricted_tstar(rows, kinds, "coupled")
    conditionals = [t for t in (tg, tt) if t is not None]
    tax = joint / max(conditionals) if conditionals and max(conditionals) > 0 \
        else None
    return CouplingReport(tstar_joint=joint, tstar_graph_only=tg,
                          tstar_target_only=tt, tstar_coupled_only=tc,
                          tax=tax)
