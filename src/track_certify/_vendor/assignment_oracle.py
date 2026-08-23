"""Exact target-assignment oracle for joint graph--target certification.

For a fixed graph, each row is an intervention environment and each column is
a candidate target.  The certification loss is additive across rows and the
main model requires distinct targets, so the best target vector is a
rectangular linear assignment.  No hypothesis is pruned.
"""

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignmentSolution:
    cost: float
    targets: Tuple[int, ...]


def _checked_cost(cost: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError("cost must be a K-by-d matrix")
    k, d = arr.shape
    if k == 0 or k > d:
        raise ValueError("distinct-target assignment requires 1 <= K <= d")
    if np.isnan(arr).any():
        raise ValueError("cost contains NaN")
    return arr


def best_assignment(
    cost: np.ndarray,
    forbidden_pairs: Iterable[Tuple[int, int]] = (),
) -> Optional[AssignmentSolution]:
    """Return the minimum-cost injective target vector, or None if infeasible."""

    arr = _checked_cost(cost).copy()
    for e, target in forbidden_pairs:
        if not (0 <= e < arr.shape[0] and 0 <= target < arr.shape[1]):
            raise ValueError("forbidden pair is outside the cost matrix")
        arr[e, target] = np.inf
    try:
        rows, cols = linear_sum_assignment(arr)
    except ValueError:
        return None
    if len(rows) != arr.shape[0] or not np.isfinite(arr[rows, cols]).all():
        return None
    targets = np.empty(arr.shape[0], dtype=int)
    targets[rows] = cols
    return AssignmentSolution(float(arr[rows, cols].sum()), tuple(targets.tolist()))


def best_distinct_assignment(
    cost: np.ndarray,
    incumbent_targets: Sequence[int],
) -> Optional[AssignmentSolution]:
    """Return the best injective assignment different from the incumbent.

    Every different assignment omits at least one incumbent row--target pair.
    Solving K assignments, each forbidding one such pair, is therefore exact.
    """

    arr = _checked_cost(cost)
    incumbent = tuple(int(t) for t in incumbent_targets)
    if len(incumbent) != arr.shape[0] or len(set(incumbent)) != len(incumbent):
        raise ValueError("incumbent_targets must be an injective K-vector")
    if any(t < 0 or t >= arr.shape[1] for t in incumbent):
        raise ValueError("incumbent target is outside the cost matrix")

    best: Optional[AssignmentSolution] = None
    for e, target in enumerate(incumbent):
        candidate = best_assignment(arr, forbidden_pairs=((e, target),))
        if candidate is not None and (best is None or candidate.cost < best.cost):
            best = candidate
    return best


def best_wrong_hypothesis(
    graph_costs: Dict[Hashable, np.ndarray],
    incumbent_graph: Hashable,
    incumbent_targets: Sequence[int],
) -> Tuple[Hashable, AssignmentSolution]:
    """Return the exact lowest-loss graph--target answer excluding incumbent."""

    if incumbent_graph not in graph_costs:
        raise ValueError("incumbent graph is absent from graph_costs")
    answer = None
    for graph, cost in graph_costs.items():
        if graph == incumbent_graph:
            solution = best_distinct_assignment(cost, incumbent_targets)
        else:
            solution = best_assignment(cost)
        if solution is not None and (answer is None or solution.cost < answer[1].cost):
            answer = (graph, solution)
    if answer is None:
        raise ValueError("the declared class contains no alternative hypothesis")
    return answer
