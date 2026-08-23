"""track-certify: anytime joint certification of causal graphs and unknown
intervention targets.

Given a declared finite class of candidate causal graphs over a shared
covariance, and K interventional environments whose targets and amplitudes
are unknown, this package answers three questions with guarantees:

* **Where to sample next, and when to stop.** :class:`Certifier` streams:
  it emits the next environment to sample, consumes one fresh observation at
  a time, and stops with a jointly certified (graph, target-vector) answer
  whose error probability is at most ``delta`` under optional stopping -- or
  with a refusal, never a guess.
* **How hard the problem is, before sampling.** :func:`characteristic_time`
  returns the instance's first-order sample requirement ``T*`` and the
  optimal budget split across environments; :func:`coupling_tax` quantifies
  how much harder joint certification is than its staged relaxations.
* **What survives an estimated covariance.** :func:`certified_envelope` and
  :func:`robust_certifier` implement the split-sample workflow with fully
  explicit constants, refusing before starting when the envelope is
  infeasible.

Ready-made benchmark instances with theorem-known difficulty are in
:func:`solvable_instance` and :func:`scalable_instance`; the simulation
driver used for the paper's experiments is exposed as :func:`simulate`.

``tests/test_equivalence.py`` proves trajectory equivalence between the
streaming interface and that simulation driver.
"""

from .api import (Certifier, Envelope, Refusal, StepOutcome, build_model,
                  certified_envelope)
from .design import (CouplingReport, DesignReport, characteristic_time,
                     coupling_tax)
from .instances import Instance, scalable_instance, solvable_instance
from .robust import RobustSetup, estimate_covariance, robust_certifier
from . import _vendor  # noqa: F401
from track_and_certify_general import run_general as simulate  # noqa: E402

__all__ = [
    "Certifier", "Envelope", "Refusal", "StepOutcome", "build_model",
    "certified_envelope",
    "DesignReport", "CouplingReport", "characteristic_time", "coupling_tax",
    "Instance", "solvable_instance", "scalable_instance",
    "RobustSetup", "estimate_covariance", "robust_certifier",
    "simulate",
]
__version__ = "0.1.0"
