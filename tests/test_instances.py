"""Instance generators and simulate round-trips."""
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
