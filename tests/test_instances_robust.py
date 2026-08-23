"""Instance generators, robust workflow, simulate round-trips."""
import numpy as np
import pytest

import track_certify as tc


def test_solvable_instance_certifies():
    inst = tc.solvable_instance(0.8)
    res = tc.simulate(inst.model, inst.truth, inst.amplitudes, 1e-2,
                      rng=np.random.default_rng(1), tmax=20_000)
    assert res.stopped and res.correct


def test_scalable_instance_certifies():
    inst = tc.scalable_instance([0.75, 0.85], [1.2, 1.2])
    res = tc.simulate(inst.model, inst.truth, inst.amplitudes, 1e-1,
                      rng=np.random.default_rng(2), tmax=60_000)
    assert res.stopped and res.correct


def test_instance_validation():
    with pytest.raises(ValueError):
        tc.solvable_instance(0.5)              # below 1/sqrt(2)
    with pytest.raises(ValueError):
        tc.solvable_instance(1.0)
    with pytest.raises(ValueError):
        tc.solvable_instance(0.9, c1=0.0)
    with pytest.raises(ValueError):
        tc.scalable_instance([0.8], [1.0])     # m < 2
    with pytest.raises(ValueError):
        tc.scalable_instance([0.8, 0.5], [1.0, 1.0])


def test_robust_workflow_feasible_and_refusing():
    rng = np.random.default_rng(3)
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    obs = rng.multivariate_normal(np.zeros(2), sigma, size=400_000)
    setup = tc.robust_certifier(obs, {"A": (0, 1), "B": (1, 0)},
                                n_environments=2, delta=0.01, c_max=3.0)
    assert setup.envelope.beta_scale > 1.0
    assert setup.certifier is not None
    assert setup.sigma_hat.shape == (2, 2)
    # infeasible n0 -> pre-start refusal, before touching the data content
    with pytest.raises(tc.Refusal):
        tc.robust_certifier(obs[:50], {"A": (0, 1), "B": (1, 0)},
                            n_environments=2, delta=0.01, c_max=3.0)


def test_estimate_covariance_center_warns():
    x = np.random.default_rng(4).normal(size=(100, 2))
    with pytest.warns(UserWarning):
        tc.estimate_covariance(x, center=True)
    with pytest.raises(ValueError):
        tc.estimate_covariance(x[:1])
