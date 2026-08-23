"""Finite Gaussian response-ray models for Track-and-Certify.

Each candidate graph is a complete DAG induced by a declared topological order.
All candidates therefore belong to the same complete-skeleton MEC and share one
known covariance matrix.  A Cholesky column is a structural propagation vector
times a target-specific residual standard deviation; because amplitudes are
profiled, it represents exactly the same response ray.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from best_alternative import Hypothesis


@dataclass(frozen=True)
class FiniteRayModel:
    sigma: np.ndarray
    graph_orders: Mapping[str, Tuple[int, ...]]
    directions: Mapping[str, np.ndarray]

    @property
    def dimension(self) -> int:
        return int(self.sigma.shape[0])

    @property
    def graph_labels(self) -> Tuple[str, ...]:
        return tuple(self.graph_orders)

    @property
    def target_labels(self) -> Tuple[int, ...]:
        return tuple(range(self.dimension))


def correlation_matrix(dimension: int, rng: np.random.Generator,
                       ridge: float) -> np.ndarray:
    factor = rng.normal(size=(dimension, dimension))
    covariance = factor @ factor.T + ridge * np.eye(dimension)
    scale = 1.0 / np.sqrt(np.diag(covariance))
    return covariance * np.outer(scale, scale)


def rays_for_order(sigma: np.ndarray, order: Sequence[int]) -> np.ndarray:
    dimension = sigma.shape[0]
    order = tuple(int(node) for node in order)
    if sorted(order) != list(range(dimension)):
        raise ValueError("order must be a permutation of all nodes")
    cholesky = np.linalg.cholesky(sigma[np.ix_(order, order)])
    directions = np.zeros((dimension, dimension))
    for position, node in enumerate(order):
        column = np.zeros(dimension)
        column[list(order)] = cholesky[:, position]
        directions[:, node] = column
    return directions


def model_from_orders(sigma: np.ndarray,
                      graph_orders: Mapping[str, Sequence[int]]) -> FiniteRayModel:
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be square")
    if not np.allclose(sigma, sigma.T, atol=1e-12, rtol=1e-12):
        raise ValueError("sigma must be symmetric")
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("sigma must be positive definite")
    if not graph_orders:
        raise ValueError("at least one graph is required")
    normalized = {
        str(graph): tuple(int(node) for node in order)
        for graph, order in graph_orders.items()
    }
    directions = {
        graph: rays_for_order(sigma, order)
        for graph, order in normalized.items()
    }
    return FiniteRayModel(sigma=sigma, graph_orders=normalized, directions=directions)


def profiled_divergence(mu: np.ndarray, ray: np.ndarray,
                        sigma_inv: np.ndarray) -> float:
    a = mu @ sigma_inv @ mu
    b = mu @ sigma_inv @ ray
    c = ray @ sigma_inv @ ray
    return float(0.5 * (a - b * b / c))


def true_means(model: FiniteRayModel, true_hypothesis: Hypothesis,
               amplitudes: Sequence[float]) -> Tuple[np.ndarray, ...]:
    graph, targets = true_hypothesis
    amplitudes = np.asarray(amplitudes, dtype=float)
    if graph not in model.directions:
        raise ValueError("true graph is absent from the declared class")
    if amplitudes.shape != (len(targets),):
        raise ValueError("one amplitude is required per environment")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be injective")
    return tuple(
        amplitudes[e] * model.directions[graph][:, target]
        for e, target in enumerate(targets)
    )


def alternative_kind(candidate: Hypothesis, truth: Hypothesis) -> str:
    graph_changed = candidate[0] != truth[0]
    targets_changed = tuple(candidate[1]) != tuple(truth[1])
    if graph_changed and targets_changed:
        return "coupled"
    if graph_changed:
        return "graph"
    if targets_changed:
        return "target"
    return "truth"


def alternative_divergences(model: FiniteRayModel, true_hypothesis: Hypothesis,
                            amplitudes: Sequence[float]):
    means = true_means(model, true_hypothesis, amplitudes)
    sigma_inv = np.linalg.inv(model.sigma)
    n_environments = len(true_hypothesis[1])
    rows = []
    kinds = []
    hypotheses = []
    for graph in model.graph_labels:
        for targets in itertools.permutations(model.target_labels, n_environments):
            candidate = (graph, tuple(targets))
            if candidate == true_hypothesis:
                continue
            rows.append(
                [
                    profiled_divergence(
                        means[e], model.directions[graph][:, targets[e]], sigma_inv
                    )
                    for e in range(n_environments)
                ]
            )
            kinds.append(alternative_kind(candidate, true_hypothesis))
            hypotheses.append(candidate)
    return np.asarray(rows), tuple(kinds), tuple(hypotheses)


def characteristic_time(divergences: np.ndarray):
    divergences = np.asarray(divergences, dtype=float)
    if divergences.ndim != 2 or divergences.shape[0] == 0:
        return np.inf, None
    n_alternatives, n_environments = divergences.shape
    objective = np.zeros(n_environments + 1)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=np.hstack([-divergences, np.ones((n_alternatives, 1))]),
        b_ub=np.zeros(n_alternatives),
        A_eq=np.array([[*([1.0] * n_environments), 0.0]]),
        b_eq=[1.0],
        bounds=[(0.0, None)] * n_environments + [(None, None)],
        method="highs",
    )
    if not result.success or result.x[-1] <= 1e-12:
        return np.inf, None
    return float(1.0 / result.x[-1]), tuple(float(x) for x in result.x[:-1])


def diagnose_characteristic_times(model: FiniteRayModel,
                                  true_hypothesis: Hypothesis,
                                  amplitudes: Sequence[float]):
    divergences, kinds, _ = alternative_divergences(
        model, true_hypothesis, amplitudes
    )
    output = {}
    for kind in ("graph", "target", "coupled"):
        mask = np.asarray([label == kind for label in kinds])
        time, weights = characteristic_time(divergences[mask])
        output[kind] = {"tstar": time, "weights": weights}
    time, weights = characteristic_time(divergences)
    output["all"] = {"tstar": time, "weights": weights}
    output["gamma_uniform"] = float(
        np.min(divergences @ (np.ones(divergences.shape[1]) / divergences.shape[1]))
    )
    return output


def graph_cost_matrices(model: FiniteRayModel, means: Sequence[np.ndarray]):
    sigma_inv = np.linalg.inv(model.sigma)
    costs: Dict[str, np.ndarray] = {}
    for graph in model.graph_labels:
        matrix = np.empty((len(means), model.dimension))
        for e, mu in enumerate(means):
            for target in model.target_labels:
                matrix[e, target] = profiled_divergence(
                    mu, model.directions[graph][:, target], sigma_inv
                )
        costs[graph] = matrix
    return costs
