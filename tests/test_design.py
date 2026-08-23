"""Design tools agree with closed forms and independent references."""
import numpy as np
import pytest

import track_certify as tc


def test_characteristic_time_matches_closed_form_solvable():
    for r, c1, c2 in [(0.75, 1.0, 1.0), (0.9, 1.0, 0.8), (0.95, 1.5, 1.0)]:
        inst = tc.solvable_instance(r, c1, c2)
        rep = tc.characteristic_time(inst.model, inst.truth, inst.amplitudes)
        assert rep.tstar == pytest.approx(inst.tstar_closed_form, rel=1e-9)
        assert np.allclose(rep.w_star, inst.w_star_closed_form, atol=1e-6)
        assert "coupled" in rep.active_kinds and "target" in rep.active_kinds
        assert rep.budget(0.01) == pytest.approx(
            inst.tstar_closed_form * np.log(100), rel=1e-12)


def test_characteristic_time_matches_closed_form_scalable():
    inst = tc.scalable_instance([0.8, 0.9], [1.0, 1.2])
    rep = tc.characteristic_time(inst.model, inst.truth, inst.amplitudes)
    assert rep.tstar == pytest.approx(inst.tstar_closed_form, rel=1e-9)
    assert np.allclose(rep.w_star, inst.w_star_closed_form, atol=1e-6)
    assert rep.active_kinds == ("coupled",)


def test_coupling_tax_two_node_law():
    # Paper eq. (21): tax = r^2/(1-r^2) for r > 1/sqrt(2), K=1 embedded as K=2
    # two-node model with equal amplitudes.
    r = 0.9
    sigma = np.array([[1.0, r], [r, 1.0]])
    model = tc.build_model(sigma, {"G12": (0, 1), "G21": (1, 0)})
    tax = tc.coupling_tax(model, ("G12", (0, 1)), (1.0, 1.0))
    assert tax.tax is not None and tax.tax > 1.0
    assert tax.tstar_joint >= max(tax.tstar_graph_only,
                                  tax.tstar_target_only)


def test_tax_reference_three_node_instance():
    # Three-node reference instance (r,s,q)=(0.85,0.45,0.0), truth graph 012,
    # targets (2,1), amplitudes (1.0, 2.8284...); independently computed
    # reference values: joint T* = 2.900900..., conditional T* (graph
    # 0.37816, target 2.25), tax = 1.2892892892892873.
    import itertools
    sigma = np.eye(3)
    sigma[0, 1] = sigma[1, 0] = 0.85
    sigma[0, 2] = sigma[2, 0] = 0.45
    orders = {"".join(map(str, p)): p
              for p in itertools.permutations(range(3))}
    with pytest.warns(UserWarning):  # q=0: one ray-identical order pair
        model = tc.build_model(sigma, orders, on_indistinct="warn")
    truth, amps = ("012", (2, 1)), (1.0, 2.8284271247461903)
    rep = tc.characteristic_time(model, truth, amps)
    assert rep.tstar == pytest.approx(2.900900900900896, rel=1e-9)
    tax = tc.coupling_tax(model, truth, amps)
    assert tax.tstar_graph_only == pytest.approx(0.3781558374983982, rel=1e-9)
    assert tax.tstar_target_only == pytest.approx(2.25, rel=1e-9)
    assert tax.tax == pytest.approx(1.2892892892892873, rel=1e-9)


def test_design_input_validation():
    inst = tc.solvable_instance(0.9)
    with pytest.raises(ValueError):
        tc.characteristic_time(inst.model, inst.truth, (1.0,))  # wrong K
    with pytest.raises(ValueError):
        tc.characteristic_time(inst.model, inst.truth, (0.0, 1.0))
    with pytest.raises(ValueError):
        tc.characteristic_time(inst.model, ("nope", (0, 2)), (1.0, 1.0))
    with pytest.raises(ValueError):
        tc.characteristic_time(inst.model, inst.truth, (1.0, 1.0)).budget(1.5)
