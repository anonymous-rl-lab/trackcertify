"""track-certify: anytime joint certification of causal graphs and unknown
intervention targets.

Given a declared finite class of candidate causal graphs over a shared
covariance, and K interventional environments whose targets and amplitudes
are unknown, this package answers two questions with guarantees:

* **Where to sample next, and when to stop.** :class:`Certifier` streams:
  it emits the next environment to sample, consumes one fresh observation at
  a time, and stops with a jointly certified (graph, target-vector) answer
  whose error probability is at most ``delta`` under optional stopping -- or
  with a refusal, never a guess.
* **How hard the problem is, before sampling.** :func:`characteristic_time`
  returns the instance's first-order sample requirement ``T*`` and the
  optimal budget split across environments; :func:`coupling_tax` quantifies
  how much harder joint certification is than its staged relaxations.

Ready-made benchmark instances with theorem-known difficulty are in
:func:`solvable_instance` and :func:`scalable_instance`; the simulation
driver used for the paper's experiments is exposed as :func:`simulate`.

The covariance is assumed known and shared. There is no estimated-covariance
entry point: the paper establishes no anytime-valid extension to an estimated
covariance, so the package offers none.

``tests/test_equivalence.py`` proves trajectory equivalence between the
streaming interface and that simulation driver.
"""

from .api import Certifier, StepOutcome, build_model
from .design import (CouplingReport, DesignReport, characteristic_time,
                     coupling_tax)
from .instances import Instance, scalable_instance, solvable_instance
from . import _vendor  # noqa: F401
from track_and_certify_general import run_general as simulate  # noqa: E402

__all__ = [
    "Certifier", "StepOutcome", "build_model",
    "DesignReport", "CouplingReport", "characteristic_time", "coupling_tax",
    "Instance", "solvable_instance", "scalable_instance",
    "simulate",
]
__version__ = "0.2.0"
