"""Streaming certification API.

`Certifier` exposes the decision loop of the simulation driver
(`_vendor/track_and_certify_general.py :: run_general`) behind a streaming
interface: the caller asks for the next environment, supplies one fresh
observation vector, and receives either nothing (continue), a certificate, or
a refusal.  The statistics, exploration floor, C-tracking, scale-free hedge
update with dyadic restarts, mixture-penalty threshold, and assignment oracle
are the same core code, called step for step in the same order; the
package's equivalence test (`tests/test_equivalence.py`) drives this class and
`run_general` over identical noise tapes and requires identical actions,
decisions, stopping times, and per-environment counts.

The certifier never receives the true hypothesis, the amplitudes, the
characteristic time, or the optimal allocation.  Exhausting the predeclared
cap is a refusal, not a low-confidence answer.

The covariance is assumed known and shared, as it is throughout the paper.
No estimated-covariance mode is offered: the paper establishes no
anytime-valid extension to an estimated covariance.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import _vendor  # noqa: F401  (installs the vendored module path)

from best_alternative import best_alternative, best_hypothesis  # noqa: E402
from finite_ray_model import FiniteRayModel, model_from_orders  # noqa: E402
from track_and_certify_general import (  # noqa: E402
    _divergence_costs,
    _hedge_weights,
    _amplitude,
    _score_costs,
)

Hypothesis = Tuple[str, Tuple[int, ...]]


@dataclass(frozen=True)
class StepOutcome:
    """Result of feeding one observation."""

    t: int
    stopped: bool
    refused: bool
    decision: Optional[Hypothesis]
    z_statistic: Optional[float]
    boundary: Optional[float]
    counts: Tuple[int, ...]

    @property
    def certificate(self) -> Optional[Hypothesis]:
        return self.decision if (self.stopped and not self.refused) else None


def build_model(sigma: np.ndarray,
                graph_orders: Dict[str, Sequence[int]],
                on_indistinct: str = "raise") -> FiniteRayModel:
    """Validate inputs and build the finite-ray model, refusing bad input.

    Refused: non-square, asymmetric, or non-positive-definite covariance;
    an empty class or empty dimension; orders that are not permutations of
    0..d-1; and duplicate order declarations. Ray-identical graph
    declarations (two distinct orders whose response-ray matrices coincide
    up to column sign) violate class-wide decision-distinctness -- no pair
    of hypotheses built from such labels can ever be certified apart, so a
    truth inside such a pair ends in a cap refusal, never a certificate.
    By default they are refused; ``on_indistinct="warn"`` downgrades this
    to a warning for declared classes whose intended truths lie outside the
    indistinct pair (e.g. boundary constructions with an exactly vanishing
    cross-link).
    """
    if on_indistinct not in {"raise", "warn"}:
        raise ValueError('on_indistinct must be "raise" or "warn"')
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or sigma.size == 0:
        raise ValueError("sigma must be a nonempty square matrix")
    if not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite")
    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("sigma must be symmetric")
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sigma must be positive definite") from exc
    if not graph_orders:
        raise ValueError("declare at least one candidate graph")
    d = sigma.shape[0]
    seen: Dict[Tuple[int, ...], str] = {}
    for label, order in graph_orders.items():
        if sorted(order) != list(range(d)):
            raise ValueError(
                f"graph {label!r}: order must be a permutation of 0..{d-1}")
        key = tuple(order)
        if key in seen:
            raise ValueError(
                f"graphs {seen[key]!r} and {label!r} declare the same order: "
                "duplicate labels violate decision-distinctness")
        seen[key] = label
    model = model_from_orders(sigma, {k: tuple(v) for k, v in
                                      graph_orders.items()})
    labels = list(model.graph_labels)
    unit = {}
    for g in labels:
        mat = np.asarray(model.directions[g], dtype=float)
        norms = np.linalg.norm(mat, axis=0)
        if np.any(norms <= 0) or not np.isfinite(mat).all():
            raise ValueError(f"graph {g!r}: degenerate response ray")
        mat = mat / norms
        # canonical column signs: rays are directions, v and -v are the same
        for j in range(mat.shape[1]):
            col = mat[:, j]
            lead = col[np.flatnonzero(np.abs(col) > 1e-12)[0]]
            if lead < 0:
                mat[:, j] = -col
        unit[g] = mat
    for i, g in enumerate(labels):
        for h in labels[i + 1:]:
            if np.allclose(unit[g], unit[h], atol=1e-10):
                msg = (f"graphs {g!r} and {h!r} induce identical response "
                       "rays: the declared class violates decision-"
                       "distinctness, and hypotheses inside this pair are "
                       "mutually uncertifiable")
                if on_indistinct == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
    return model


class Certifier:
    """Streaming Track-and-Certify.

    Usage::

        c = Certifier(model, n_environments=K, delta=0.01, cap=200_000)
        while True:
            e = c.next_environment()
            y = <one fresh observation vector from environment e>
            out = c.observe(y)
            if out.stopped:
                break
        out.certificate  # (graph, targets) or None if out.refused
    """

    def __init__(self, model: FiniteRayModel, n_environments: int,
                 delta: float, *, cap: int = 100_000, rho: float = 1.0,
                 backend: str = "assignment",
                 sampling_policy: str = "adaptive",
                 fixed_weights: Optional[Sequence[float]] = None) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        if backend not in {"assignment", "enumeration"}:
            raise ValueError("backend must be 'assignment' or 'enumeration'")
        if sampling_policy not in {"adaptive", "uniform", "oracle"}:
            raise ValueError(
                "sampling_policy must be adaptive, uniform, or oracle")
        if n_environments < 1 or n_environments > model.dimension:
            raise ValueError("need 1 <= K <= d (injective targets)")
        if not (isinstance(cap, (int, np.integer)) and cap >= 1):
            raise ValueError("cap must be a positive integer")
        if rho <= 0 or not np.isfinite(rho):
            raise ValueError("rho must be positive and finite")
        self._model = model
        self._K = int(n_environments)
        self._delta = float(delta)
        self._cap = int(cap)
        self._rho = float(rho)
        self._backend = backend
        self._policy = sampling_policy
        self._sigma_inv = np.linalg.inv(model.sigma)
        # exact response-subspace projection leaves d-1 residual dimensions
        self._pdims = model.dimension - 1.0
        if sampling_policy == "uniform":
            self._fixed = np.ones(self._K) / self._K
        elif sampling_policy == "oracle":
            if fixed_weights is None:
                raise ValueError("oracle policy requires fixed_weights")
            w = np.asarray(fixed_weights, dtype=float)
            if w.shape != (self._K,) or not np.isfinite(w).all() \
                    or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("fixed_weights must be a nonnegative "
                                 "length-K vector with positive mass")
            self._fixed = w / w.sum()
        else:
            self._fixed = None
        # dynamic state (mirrors run_general step for step)
        self._t = 0
        self._sums = [np.zeros(model.dimension) for _ in range(self._K)]
        self._counts = [0] * self._K
        self._gain = np.zeros(self._K)
        self._gain_sq = 0.0
        self._last_hypothesis: Optional[Hypothesis] = None
        self._cumulative = np.zeros(self._K)
        self._pending_action: Optional[int] = None
        self._done = False

    # -- streaming interface ------------------------------------------------
    def next_environment(self) -> int:
        """Choose the environment for step t+1 (floor + C-tracking)."""
        if self._done:
            raise RuntimeError("certification has terminated")
        if self._pending_action is not None:
            return self._pending_action
        t = self._t + 1
        if self._policy == "adaptive" and t > 1 and (t & (t - 1)) == 0:
            self._gain = np.zeros(self._K)
            self._gain_sq = 0.0
        if self._policy == "adaptive":
            w = _hedge_weights(self._gain, self._gain_sq)
            gamma = 1.0 / (2.0 * np.sqrt(t))
            floored = (1.0 - gamma) * w + gamma / self._K
        else:
            floored = self._fixed
        self._cumulative += floored
        action = int(np.argmax(self._cumulative - np.asarray(self._counts)))
        self._pending_action = action
        return action

    def observe(self, observation: Sequence[float]) -> StepOutcome:
        """Consume one fresh observation from the environment last returned."""
        if self._pending_action is None:
            raise RuntimeError("call next_environment() before observe()")
        y = np.asarray(observation, dtype=float)
        if y.shape != (self._model.dimension,) or not np.isfinite(y).all():
            raise ValueError("observation must be a finite length-d vector")
        action = self._pending_action
        self._pending_action = None
        self._t += 1
        t = self._t
        self._sums[action] += y
        self._counts[action] += 1

        if min(self._counts) < 2:
            return self._maybe_cap(StepOutcome(
                t, False, False, None, None, None, tuple(self._counts)))

        bars = [self._sums[e] / self._counts[e] for e in range(self._K)]
        score_costs = _score_costs(self._model, bars, self._counts,
                                   self._rho, self._sigma_inv)
        incumbent = best_hypothesis(score_costs,
                                    target_labels=self._model.target_labels,
                                    backend=self._backend)
        stopping_alt = best_alternative(
            score_costs, incumbent.hypothesis,
            target_labels=self._model.target_labels, backend=self._backend)
        decision = incumbent.hypothesis
        z = stopping_alt.cost - incumbent.cost
        penalty = self._pdims / 2.0 * sum(
            np.log1p(c / self._rho) for c in self._counts)
        boundary = np.log(1.0 / self._delta) + penalty
        if not np.isfinite([z, boundary]).all():
            raise FloatingPointError("non-finite stopping state")
        if z >= boundary:
            self._done = True
            return StepOutcome(t, True, False, decision, float(z),
                               float(boundary), tuple(self._counts))

        if self._policy == "adaptive":
            if self._last_hypothesis is None or decision != self._last_hypothesis:
                self._gain = np.zeros(self._K)
                self._gain_sq = 0.0
                self._last_hypothesis = decision
            amps = [_amplitude(
                bars[e],
                self._model.directions[decision[0]][:, decision[1][e]],
                self._sigma_inv) for e in range(self._K)]
            means = [amps[e]
                     * self._model.directions[decision[0]][:, decision[1][e]]
                     for e in range(self._K)]
            game_w = _hedge_weights(self._gain, self._gain_sq)
            div_costs = _divergence_costs(self._model, means, self._sigma_inv)
            nature = best_alternative(
                div_costs, decision, environment_weights=game_w,
                target_labels=self._model.target_labels,
                backend=self._backend)
            reward = np.asarray(nature.row_costs)
            if not np.isfinite(reward).all():
                raise FloatingPointError("non-finite allocation reward")
            self._gain += reward
            self._gain_sq += float(np.max(np.abs(reward)) ** 2)

        return self._maybe_cap(StepOutcome(
            t, False, False, decision, float(z), float(boundary),
            tuple(self._counts)))

    def _maybe_cap(self, out: StepOutcome) -> StepOutcome:
        if self._t >= self._cap:
            self._done = True
            return StepOutcome(out.t, True, True, out.decision,
                               out.z_statistic, out.boundary, out.counts)
        return out
