"""Estimated-covariance certification workflow.

When the observational covariance is not known, this module estimates
``Sigma`` once from ``n0`` independent observational samples, applies an
explicit spectral envelope to that estimate, and runs the certifier with the
correspondingly inflated threshold.  When the envelope's computable
feasibility conditions fail -- decidable from ``(n0, delta, d, c_max)``
before any interventional data are drawn -- the workflow *refuses to start*
instead of certifying.

**This path carries no validity theorem.**  The accompanying paper assumes a
known common covariance throughout.  Its supplement, in "Estimated
Covariance: Required Conditions", lists what an anytime-valid extension would
have to supply at once -- a high-probability spectral envelope uniform over
the candidate response subspaces, an amplitude envelope controlling
response-direction perturbation, a boundary that stays valid after covariance
uncertainty is propagated through the adaptive e-process, and an
allocation-game perturbation bound preserving positive separation -- and
states that none of them is established there.  What follows is therefore a
self-contained construction with an explicit refusal gate, not a certified
error guarantee, and the error level it runs at is not covered by the
paper's Theorem 1(ii).

The estimator the constants are written for is the zero-mean second-moment
estimator ``X'X / n0`` for centered observational data; ``center=True``
demeans first and is provided for convenience, outside the letter of those
constants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .api import (Certifier, Envelope, Refusal, build_model,
                  certified_envelope)
from . import _vendor  # noqa: F401

from finite_ray_model import FiniteRayModel  # noqa: E402


@dataclass(frozen=True)
class RobustSetup:
    """Everything the estimated-covariance pipeline deploys.

    ``certifier`` is ready to stream (its threshold already carries the
    envelope inflation and drift, run at level ``delta/2`` with the
    full-dimension penalty); ``sigma_hat`` and ``envelope`` document what was
    estimated and what was assumed.
    """

    certifier: Certifier
    model: FiniteRayModel
    sigma_hat: np.ndarray
    envelope: Envelope
    n0: int


def estimate_covariance(observations: np.ndarray,
                        center: bool = False) -> np.ndarray:
    """Second-moment covariance estimate from observational samples.

    ``observations`` is an ``(n0, d)`` array of independent draws with mean
    zero under the model; ``center=True`` subtracts the sample mean first
    (outside the letter of the envelope constants -- a warning is issued).
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("observations must be a (n0, d) array with n0 >= 2")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite")
    if center:
        warnings.warn(
            "center=True demeans before estimating; the envelope constants "
            "are written for the zero-mean estimator X'X/n0",
            UserWarning, stacklevel=2)
        x = x - x.mean(axis=0, keepdims=True)
    return x.T @ x / x.shape[0]


def robust_certifier(observations: np.ndarray,
                     graph_orders: Dict[str, Sequence[int]],
                     n_environments: int,
                     delta: float,
                     c_max: float,
                     *,
                     gamma: float = 0.3,
                     cap: int = 100_000,
                     rho: float = 1.0,
                     center: bool = False,
                     on_indistinct: str = "raise") -> RobustSetup:
    """One call from observational samples to a deployable robust certifier.

    Raises :class:`track_certify.Refusal` when the envelope's feasibility
    conditions fail for this ``(n0, delta, d, c_max)``, before any
    interventional data are drawn, and ``ValueError`` on malformed input.
    See the module docstring: this workflow is not covered by the paper's
    validity theorem.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2:
        raise ValueError("observations must be a (n0, d) array")
    n0, d = x.shape
    envelope = certified_envelope(n0=n0, delta=delta, d=d, c_max=c_max,
                                  gamma=gamma)
    sigma_hat = estimate_covariance(x, center=center)
    model = build_model(sigma_hat, graph_orders, on_indistinct=on_indistinct)
    certifier = Certifier(
        model, n_environments=n_environments, delta=envelope.level,
        cap=cap, rho=rho, penalty_dims=envelope.penalty_dims,
        beta_scale=envelope.beta_scale, beta_drift=envelope.beta_drift)
    return RobustSetup(certifier=certifier, model=model,
                       sigma_hat=sigma_hat, envelope=envelope, n0=n0)


__all__ = ["RobustSetup", "estimate_covariance", "robust_certifier",
           "Refusal"]
