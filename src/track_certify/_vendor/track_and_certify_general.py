"""General finite-ray Track-and-Certify engine.

This module implements the certifier for declared
finite graph classes with arbitrary d and K <= d.  It preserves the same
profiled statistic, e-process boundary, scale-free Hedge, vanishing floor, and
cumulative C-tracking.  Production optimization is assignment based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from best_alternative import Hypothesis, best_alternative, best_hypothesis
from finite_ray_model import FiniteRayModel, true_means


@dataclass(frozen=True)
class GeneralStepRecord:
    t: int
    action: int
    counts: Tuple[int, ...]
    sampling_weights: Tuple[float, ...]
    floored_weights: Tuple[float, ...]
    incumbent: Optional[Hypothesis]
    stopping_alternative: Optional[Hypothesis]
    nature_alternative: Optional[Hypothesis]
    game_weights: Optional[Tuple[float, ...]]
    z_statistic: Optional[float]
    boundary: Optional[float]
    log_e_value: Optional[float]
    stopped: bool


@dataclass(frozen=True)
class GeneralRunResult:
    tau: int
    stopped: bool
    correct: bool
    decision: Optional[Hypothesis]
    counts: Tuple[int, ...]
    max_tracking_deficit: float
    trace: Tuple[GeneralStepRecord, ...]


def generate_noise_tape(seed: int, n_environments: int, max_pulls: int,
                        dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_environments, max_pulls, dimension))


def _resid2(y, ray, sigma_inv):
    a = y @ sigma_inv @ y
    b = y @ sigma_inv @ ray
    c = ray @ sigma_inv @ ray
    return float(a - b * b / c)


def _amplitude(y, ray, sigma_inv):
    return float((y @ sigma_inv @ ray) / (ray @ sigma_inv @ ray))


def _hedge_weights(gain, gain_sq):
    n_environments = len(gain)
    if gain_sq <= 1e-15:
        return np.ones(n_environments) / n_environments
    eta = np.sqrt(2.0 * np.log(n_environments) / (gain_sq + 1e-15))
    logits = eta * gain
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def _score_costs(model, bars, counts, rho, sigma_inv):
    costs = {}
    for graph in model.graph_labels:
        matrix = np.empty((len(counts), model.dimension))
        for e, count in enumerate(counts):
            coefficient = count ** 2 / (2.0 * (count + rho))
            for target in model.target_labels:
                matrix[e, target] = coefficient * _resid2(
                    bars[e], model.directions[graph][:, target], sigma_inv
                )
        costs[graph] = matrix
    return costs


def _divergence_costs(model, means, sigma_inv):
    costs = {}
    for graph in model.graph_labels:
        matrix = np.empty((len(means), model.dimension))
        for e, mean in enumerate(means):
            for target in model.target_labels:
                matrix[e, target] = 0.5 * _resid2(
                    mean, model.directions[graph][:, target], sigma_inv
                )
        costs[graph] = matrix
    return costs


def run_general(
    model: FiniteRayModel,
    true_hypothesis: Hypothesis,
    amplitudes: Sequence[float],
    delta: float,
    *,
    backend: str = "assignment",
    sampling_policy: str = "adaptive",
    oracle_weights: Optional[Sequence[float]] = None,
    rng: Optional[np.random.Generator] = None,
    noise_tape: Optional[np.ndarray] = None,
    tmax: int = 3000,
    rho: float = 1.0,
    record_trace: bool = False,
    truth_model: Optional[FiniteRayModel] = None,
    penalty_dims: Optional[int] = None,
    beta_scale: float = 1.0,
    beta_drift: float = 0.0,
) -> GeneralRunResult:
    if (rng is None) == (noise_tape is None):
        raise ValueError("supply exactly one of rng or noise_tape")
    if backend not in {"assignment", "enumeration"}:
        raise ValueError("backend must be 'assignment' or 'enumeration'")
    if sampling_policy not in {"adaptive", "uniform", "oracle", "dtracking"}:
        raise ValueError(
            "sampling_policy must be adaptive, uniform, oracle, or dtracking")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")

    # Estimated-covariance certification (Theorem 4) parameters:
    #   truth_model  -- model generating the DATA (true covariance and rays);
    #                   decisions, rays, whitening, and statistics still come
    #                   from `model`. Defaults to `model` (well-specified case).
    #   penalty_dims -- residual dimensions per environment in the mixture
    #                   penalty. Defaults to d-1 (exact projection); the
    #                   robust rule uses d (no exact projection available).
    #   beta_scale   -- multiplicative threshold inflation (1+gamma)(1+eps).
    #   beta_drift   -- additive per-sample drift term (1+1/gamma) eta_bar^2/2.
    # Defaults reproduce the known-covariance rule exactly.
    if truth_model is None:
        truth_model = model
    dimension = model.dimension
    n_environments = len(true_hypothesis[1])
    fixed_weights = None
    if sampling_policy == "uniform":
        fixed_weights = np.ones(n_environments) / n_environments
    elif sampling_policy == "oracle":
        if oracle_weights is None:
            raise ValueError("oracle policy requires oracle_weights")
        fixed_weights = np.asarray(oracle_weights, dtype=float)
        if fixed_weights.shape != (n_environments,):
            raise ValueError("oracle_weights must have length K")
        if not np.isfinite(fixed_weights).all() or np.any(fixed_weights < 0):
            raise ValueError("oracle_weights must be finite and nonnegative")
        if fixed_weights.sum() <= 0:
            raise ValueError("oracle_weights must have positive total mass")
        fixed_weights = fixed_weights / fixed_weights.sum()
    means = true_means(truth_model, true_hypothesis, amplitudes)
    sigma_inv = np.linalg.inv(model.sigma)
    cholesky = np.linalg.cholesky(truth_model.sigma)

    if noise_tape is not None:
        noise_tape = np.asarray(noise_tape, dtype=float)
        if noise_tape.ndim != 3:
            raise ValueError(
                "noise_tape must have shape (K, max_pulls, dimension)"
            )
        expected = (n_environments, noise_tape.shape[1], dimension)
        if noise_tape.shape != expected:
            raise ValueError(
                "noise_tape must have shape (K, max_pulls, dimension)"
            )

    # Plug-in D-tracking comparator.  It replaces the
    # no-regret game with direct tracking of the plug-in optimal allocation
    # w*(theta_hat_t), obtained by solving the characteristic-time LP at the
    # profiled estimate after each observation.  The stopping rule, floor, and
    # C-tracking are identical to the adaptive path; only the allocation target
    # differs.
    plugin_weights = np.ones(n_environments) / n_environments
    if sampling_policy == "dtracking":
        from finite_ray_model import (
            alternative_divergences as _alt_div,
            characteristic_time as _char_time,
        )

    sums = [np.zeros(dimension) for _ in range(n_environments)]
    counts = [0] * n_environments
    gain = np.zeros(n_environments)
    gain_sq = 0.0
    last_hypothesis = None
    cumulative_weights = np.zeros(n_environments)
    max_deficit = 0.0
    trace: List[GeneralStepRecord] = []
    current_decision = None

    for t in range(1, tmax + 1):
        if sampling_policy == "adaptive" and t > 1 and (t & (t - 1)) == 0:
            gain = np.zeros(n_environments)
            gain_sq = 0.0

        if sampling_policy == "adaptive":
            sampling_weights = _hedge_weights(gain, gain_sq)
            gamma = 1.0 / (2.0 * np.sqrt(t))
            floored_weights = (
                (1.0 - gamma) * sampling_weights + gamma / n_environments
            )
        elif sampling_policy == "dtracking":
            sampling_weights = plugin_weights
            gamma = 1.0 / (2.0 * np.sqrt(t))
            floored_weights = (
                (1.0 - gamma) * sampling_weights + gamma / n_environments
            )
        else:
            assert fixed_weights is not None
            sampling_weights = fixed_weights
            floored_weights = fixed_weights
        cumulative_weights += floored_weights
        action = int(np.argmax(cumulative_weights - np.asarray(counts)))

        pull_index = counts[action]
        if noise_tape is not None:
            if pull_index >= noise_tape.shape[1]:
                raise ValueError("noise_tape is shorter than the trajectory")
            innovation = noise_tape[action, pull_index]
        else:
            assert rng is not None
            innovation = rng.normal(size=dimension)
        observation = innovation @ cholesky.T + means[action]
        if not np.isfinite(observation).all():
            raise FloatingPointError("non-finite observation")
        sums[action] += observation
        counts[action] += 1
        max_deficit = max(
            max_deficit,
            float(np.max(np.abs(np.asarray(counts) - cumulative_weights))),
        )

        if min(counts) < 2:
            if record_trace:
                trace.append(
                    GeneralStepRecord(
                        t, action, tuple(counts), tuple(sampling_weights),
                        tuple(floored_weights), None, None, None, None,
                        None, None, None, False,
                    )
                )
            continue

        bars = [sums[e] / counts[e] for e in range(n_environments)]
        score_costs = _score_costs(model, bars, counts, rho, sigma_inv)
        incumbent = best_hypothesis(
            score_costs,
            target_labels=model.target_labels,
            backend=backend,
        )
        stopping_alt = best_alternative(
            score_costs,
            incumbent.hypothesis,
            target_labels=model.target_labels,
            backend=backend,
        )
        current_decision = incumbent.hypothesis
        z_statistic = stopping_alt.cost - incumbent.cost
        pdims = (dimension - 1.0) if penalty_dims is None else float(penalty_dims)
        penalty = pdims / 2.0 * sum(
            np.log1p(count / rho) for count in counts
        )
        boundary = beta_scale * (np.log(1.0 / delta) + penalty) + beta_drift * t
        log_e_value = z_statistic - penalty
        if not np.isfinite([z_statistic, boundary, log_e_value]).all():
            raise FloatingPointError("non-finite stopping state")
        stopped = bool(z_statistic >= boundary)

        if stopped:
            if record_trace:
                trace.append(
                    GeneralStepRecord(
                        t, action, tuple(counts), tuple(sampling_weights),
                        tuple(floored_weights), current_decision,
                        stopping_alt.hypothesis, None, None, z_statistic,
                        boundary, log_e_value, True,
                    )
                )
            return GeneralRunResult(
                tau=t,
                stopped=True,
                correct=(current_decision == true_hypothesis),
                decision=current_decision,
                counts=tuple(counts),
                max_tracking_deficit=max_deficit,
                trace=tuple(trace),
            )

        nature_alt = None
        game_weights = None
        if sampling_policy == "adaptive":
            if last_hypothesis is None or current_decision != last_hypothesis:
                gain = np.zeros(n_environments)
                gain_sq = 0.0
                last_hypothesis = current_decision

            estimated_amplitudes = [
                _amplitude(
                    bars[e],
                    model.directions[current_decision[0]][:, current_decision[1][e]],
                    sigma_inv,
                )
                for e in range(n_environments)
            ]
            estimated_means = [
                estimated_amplitudes[e]
                * model.directions[current_decision[0]][:, current_decision[1][e]]
                for e in range(n_environments)
            ]
            game_weights = _hedge_weights(gain, gain_sq)
            divergence_costs = _divergence_costs(
                model, estimated_means, sigma_inv
            )
            nature_alt = best_alternative(
                divergence_costs,
                current_decision,
                environment_weights=game_weights,
                target_labels=model.target_labels,
                backend=backend,
            )
            reward = np.asarray(nature_alt.row_costs)
            if not np.isfinite(reward).all():
                raise FloatingPointError("non-finite allocation reward")
            gain += reward
            gain_sq += float(np.max(np.abs(reward)) ** 2)
        elif sampling_policy == "dtracking":
            estimated_amplitudes = [
                _amplitude(
                    bars[e],
                    model.directions[current_decision[0]][:, current_decision[1][e]],
                    sigma_inv,
                )
                for e in range(n_environments)
            ]
            rows, _, _ = _alt_div(model, current_decision, estimated_amplitudes)
            value, weights = _char_time(rows)
            if weights is not None and np.isfinite(value):
                plugin_weights = np.asarray(weights, dtype=float)

        if record_trace:
            trace.append(
                GeneralStepRecord(
                    t, action, tuple(counts), tuple(sampling_weights),
                    tuple(floored_weights), current_decision,
                    stopping_alt.hypothesis,
                    None if nature_alt is None else nature_alt.hypothesis,
                    None if game_weights is None else tuple(game_weights),
                    z_statistic, boundary,
                    log_e_value, False,
                )
            )

    return GeneralRunResult(
        tau=tmax,
        stopped=False,
        correct=False,
        decision=current_decision,
        counts=tuple(counts),
        max_tracking_deficit=max_deficit,
        trace=tuple(trace),
    )
