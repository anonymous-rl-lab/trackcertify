"""Unified best-hypothesis and best-alternative interface.

The sequential algorithm has two consumers of the same optimization problem:

* the stopping statistic selects the profiled incumbent and its best wrong
  graph--target answer;
* the allocation game asks nature for the best wrong answer under the current
  environment weights.

Both consumers call the functions in this module.  ``backend="enumeration"``
is an exact small-instance reference implementation.  ``backend="assignment"``
uses the rectangular assignment oracle and is the production implementation.
The returned ``row_costs`` are always *unweighted*, which lets the allocation
learner update with the full environment-wise reward vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from assignment_oracle import best_assignment, best_wrong_hypothesis


Hypothesis = Tuple[Hashable, Tuple[Hashable, ...]]


@dataclass(frozen=True)
class HypothesisSolution:
    """Exact solution returned by either backend."""

    graph: Hashable
    targets: Tuple[Hashable, ...]
    cost: float
    row_costs: Tuple[float, ...]

    @property
    def hypothesis(self) -> Hypothesis:
        return self.graph, self.targets


def _validated_problem(
    graph_costs: Mapping[Hashable, np.ndarray],
    target_labels: Optional[Sequence[Hashable]],
    environment_weights: Optional[Sequence[float]],
) -> Tuple[Dict[Hashable, np.ndarray], Tuple[Hashable, ...], np.ndarray]:
    if not graph_costs:
        raise ValueError("graph_costs must contain at least one graph")

    arrays: Dict[Hashable, np.ndarray] = {}
    shape = None
    for graph, cost in graph_costs.items():
        arr = np.asarray(cost, dtype=float)
        if arr.ndim != 2:
            raise ValueError("every graph cost must be a K-by-d matrix")
        if not np.isfinite(arr).all():
            raise ValueError("graph costs must be finite")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError("all graph cost matrices must have the same shape")
        arrays[graph] = arr

    assert shape is not None
    k, d = shape
    if k == 0 or k > d:
        raise ValueError("injective targets require 1 <= K <= d")

    labels = tuple(range(d)) if target_labels is None else tuple(target_labels)
    if len(labels) != d or len(set(labels)) != d:
        raise ValueError("target_labels must contain d distinct labels")

    if environment_weights is None:
        weights = np.ones(k, dtype=float)
    else:
        weights = np.asarray(environment_weights, dtype=float)
        if weights.shape != (k,):
            raise ValueError("environment_weights must have length K")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("environment_weights must be finite and nonnegative")
        if weights.sum() <= 0:
            raise ValueError("environment_weights must have positive total mass")
    return arrays, labels, weights


def _solution_from_columns(
    graph: Hashable,
    columns: Sequence[int],
    arrays: Mapping[Hashable, np.ndarray],
    labels: Sequence[Hashable],
    weights: np.ndarray,
) -> HypothesisSolution:
    cols = np.asarray(columns, dtype=int)
    row = arrays[graph][np.arange(len(cols)), cols]
    return HypothesisSolution(
        graph=graph,
        targets=tuple(labels[j] for j in cols),
        cost=float(weights @ row),
        row_costs=tuple(float(x) for x in row),
    )


def _lex_key(solution: HypothesisSolution, graph_order: Mapping[Hashable, int],
             label_order: Mapping[Hashable, int]) -> Tuple[int, Tuple[int, ...]]:
    return graph_order[solution.graph], tuple(label_order[t] for t in solution.targets)


def _prefer(
    candidate: HypothesisSolution,
    incumbent: Optional[HypothesisSolution],
    graph_order: Mapping[Hashable, int],
    label_order: Mapping[Hashable, int],
    tie_atol: float,
) -> bool:
    if incumbent is None:
        return True
    scale = 1.0 + max(abs(candidate.cost), abs(incumbent.cost))
    if candidate.cost < incumbent.cost - tie_atol * scale:
        return True
    if abs(candidate.cost - incumbent.cost) <= tie_atol * scale:
        return _lex_key(candidate, graph_order, label_order) < _lex_key(
            incumbent, graph_order, label_order
        )
    return False


def _enumeration_solve(
    arrays: Mapping[Hashable, np.ndarray],
    labels: Sequence[Hashable],
    weights: np.ndarray,
    excluded: Optional[Hypothesis],
    tie_atol: float,
) -> HypothesisSolution:
    graph_order = {graph: i for i, graph in enumerate(arrays)}
    label_order = {label: i for i, label in enumerate(labels)}
    best = None
    for graph, arr in arrays.items():
        for cols in itertools.permutations(range(arr.shape[1]), arr.shape[0]):
            targets = tuple(labels[j] for j in cols)
            if excluded is not None and (graph, targets) == excluded:
                continue
            candidate = _solution_from_columns(graph, cols, arrays, labels, weights)
            if np.isfinite(candidate.cost) and _prefer(
                candidate, best, graph_order, label_order, tie_atol
            ):
                best = candidate
    if best is None:
        raise ValueError("the declared class contains no feasible hypothesis")
    return best


def _assignment_incumbent(
    arrays: Mapping[Hashable, np.ndarray],
    labels: Sequence[Hashable],
    weights: np.ndarray,
    tie_atol: float,
) -> HypothesisSolution:
    graph_order = {graph: i for i, graph in enumerate(arrays)}
    label_order = {label: i for i, label in enumerate(labels)}
    best = None
    for graph, arr in arrays.items():
        assignment = best_assignment(arr * weights[:, None])
        if assignment is None:
            continue
        candidate = _solution_from_columns(
            graph, assignment.targets, arrays, labels, weights
        )
        if _prefer(candidate, best, graph_order, label_order, tie_atol):
            best = candidate
    if best is None:
        raise ValueError("the declared class contains no feasible hypothesis")
    return best


def _assignment_alternative(
    arrays: Mapping[Hashable, np.ndarray],
    labels: Sequence[Hashable],
    weights: np.ndarray,
    incumbent: Hypothesis,
) -> HypothesisSolution:
    incumbent_graph, incumbent_targets = incumbent
    if incumbent_graph not in arrays:
        raise ValueError("incumbent graph is absent from graph_costs")
    label_to_col = {label: i for i, label in enumerate(labels)}
    try:
        incumbent_columns = tuple(label_to_col[t] for t in incumbent_targets)
    except KeyError as error:
        raise ValueError("incumbent target is absent from target_labels") from error
    if len(incumbent_columns) != next(iter(arrays.values())).shape[0]:
        raise ValueError("incumbent target vector must have length K")
    if len(set(incumbent_columns)) != len(incumbent_columns):
        raise ValueError("incumbent target vector must be injective")

    weighted = {graph: arr * weights[:, None] for graph, arr in arrays.items()}
    graph, assignment = best_wrong_hypothesis(
        weighted, incumbent_graph, incumbent_columns
    )
    return _solution_from_columns(graph, assignment.targets, arrays, labels, weights)


def best_hypothesis(
    graph_costs: Mapping[Hashable, np.ndarray],
    *,
    environment_weights: Optional[Sequence[float]] = None,
    target_labels: Optional[Sequence[Hashable]] = None,
    backend: str = "assignment",
    tie_atol: float = 1e-12,
) -> HypothesisSolution:
    """Return the exact minimum-cost graph--target hypothesis."""

    arrays, labels, weights = _validated_problem(
        graph_costs, target_labels, environment_weights
    )
    if backend == "assignment":
        return _assignment_incumbent(arrays, labels, weights, tie_atol)
    if backend == "enumeration":
        return _enumeration_solve(arrays, labels, weights, None, tie_atol)
    raise ValueError("backend must be 'assignment' or 'enumeration'")


def best_alternative(
    graph_costs: Mapping[Hashable, np.ndarray],
    incumbent: Hypothesis,
    *,
    environment_weights: Optional[Sequence[float]] = None,
    target_labels: Optional[Sequence[Hashable]] = None,
    backend: str = "assignment",
    tie_atol: float = 1e-12,
) -> HypothesisSolution:
    """Return the exact best graph--target answer different from ``incumbent``."""

    arrays, labels, weights = _validated_problem(
        graph_costs, target_labels, environment_weights
    )
    normalized_incumbent = (incumbent[0], tuple(incumbent[1]))
    if backend == "assignment":
        return _assignment_alternative(arrays, labels, weights, normalized_incumbent)
    if backend == "enumeration":
        return _enumeration_solve(
            arrays, labels, weights, normalized_incumbent, tie_atol
        )
    raise ValueError("backend must be 'assignment' or 'enumeration'")
