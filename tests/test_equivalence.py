"""Equivalence: streaming Certifier == simulation driver run_general, tape for tape.

Five instances, both allocation policies (adaptive, uniform): identical
action sequences, decisions, stopping times, and per-environment counts.
"""

import itertools

import numpy as np
import pytest

import track_certify as tc
from track_and_certify_general import generate_noise_tape, run_general
from finite_ray_model import true_means


def two_block_instance(r, third=None):
    if third is None:
        sigma = np.array([[1.0, r], [r, 1.0]])
        orders = {"G12": (0, 1), "G21": (1, 0)}
        d = 2
    else:
        s, q = third
        sigma = np.eye(3)
        sigma[0, 1] = sigma[1, 0] = r
        sigma[0, 2] = sigma[2, 0] = s
        sigma[1, 2] = sigma[2, 1] = q
        orders = {"".join(map(str, p)): p
                  for p in itertools.permutations(range(3))}
        d = 3
    return sigma, orders, d


INSTANCES = [
    # (label, sigma-args, truth, amplitudes, delta)
    ("A", (0.5, None), ("G12", (0, 1)), (1.5, 1.2), 1e-2),
    ("B", (0.8, None), ("G12", (0, 1)), (1.0, 1.0), 1e-2),
    ("C", (0.3, None), ("G21", (1, 0)), (2.0, 0.8), 1e-1),
    ("D", (0.85, (0.45, 0.0)), ("012", (2, 1)), (1.0, 2.83), 1e-2),
    ("E", (0.72, (0.15, 0.45)), ("012", (2, 1)), (1.0, 1.0), 1e-1),
]


@pytest.mark.parametrize("label,sargs,truth,amps,delta", INSTANCES)
@pytest.mark.parametrize("policy", ["adaptive", "uniform"])
def test_streaming_matches_reference_driver(label, sargs, truth, amps, delta,
                                          policy):
    sigma, orders, d = two_block_instance(*sargs)
    if label == "D":  # q = 0: ray-identical pair among the declared orders
        with pytest.warns(UserWarning, match="decision-"):
            model = tc.build_model(sigma, orders, on_indistinct="warn")
    else:
        model = tc.build_model(sigma, orders)
    K = len(truth[1])
    tape = generate_noise_tape(20260823, K, 4000, d)

    ref = run_general(model, truth, amps, delta, sampling_policy=policy,
                      noise_tape=tape, tmax=3000, record_trace=True)
    assert ref.stopped, f"{label}: reference run did not stop"

    means = true_means(model, truth, amps)
    chol = np.linalg.cholesky(model.sigma)
    cert = tc.Certifier(model, n_environments=K, delta=delta,
                        sampling_policy=policy, cap=3000)
    actions, out = [], None
    pulls = [0] * K
    while True:
        e = cert.next_environment()
        actions.append(e)
        y = tape[e, pulls[e]] @ chol.T + means[e]
        pulls[e] += 1
        out = cert.observe(y)
        if out.stopped:
            break

    assert not out.refused, f"{label}: unexpected refusal"
    assert out.t == ref.tau, (label, policy, out.t, ref.tau)
    assert out.decision == ref.decision, (label, policy)
    assert out.counts == ref.counts, (label, policy)
    ref_actions = [rec.action for rec in ref.trace]
    assert actions == ref_actions, (label, policy)


def test_refusals():
    sigma, orders, d = two_block_instance(0.5, None)
    model = tc.build_model(sigma, orders)
    cert = tc.Certifier(model, n_environments=2, delta=1e-2, cap=3)
    rng = np.random.default_rng(0)
    out = None
    for _ in range(3):
        cert.next_environment()
        out = cert.observe(rng.normal(size=2))
    assert out.stopped and out.refused and out.certificate is None

    with pytest.raises(tc.Refusal):
        tc.certified_envelope(n0=200, delta=1e-2, d=3, c_max=1.0)
    env = tc.certified_envelope(n0=512_000, delta=1e-2, d=3, c_max=1.0)
    assert env.beta_scale > 1.0 and env.beta_drift > 0.0 and env.penalty_dims == 3

    with pytest.raises(ValueError):
        tc.build_model(np.array([[1.0, 2.0], [2.0, 1.0]]),
                       {"G12": (0, 1)})  # not PD


def test_input_validation():
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    # duplicate order labels violate decision-distinctness
    with pytest.raises(ValueError, match="decision-"):
        tc.build_model(sigma, {"A": (0, 1), "B": (0, 1)})
    # ray-identical declarations under distinct orders (independent nodes)
    with pytest.raises(ValueError, match="decision-"):
        tc.build_model(np.eye(2), {"A": (0, 1), "B": (1, 0)})
    with pytest.warns(UserWarning, match="decision-"):
        tc.build_model(np.eye(2), {"A": (0, 1), "B": (1, 0)},
                       on_indistinct="warn")
    with pytest.raises(ValueError):
        tc.build_model(np.eye(2), {"A": (0, 1), "B": (1, 0)},
                       on_indistinct="silent")
    # empty class / empty dimension
    with pytest.raises(ValueError):
        tc.build_model(sigma, {})
    with pytest.raises(ValueError):
        tc.build_model(np.zeros((0, 0)), {})
    # envelope parameter validation: refusals are ValueError, never
    # ZeroDivisionError or silent acceptance
    for bad in [dict(n0=0), dict(n0=-5), dict(d=0), dict(c_max=-1.0),
                dict(c_max=0.0), dict(gamma=0.0), dict(delta=0.0),
                dict(delta=1.0)]:
        kw = dict(n0=512_000, delta=1e-2, d=3, c_max=1.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            tc.certified_envelope(**kw)
    # certifier parameter validation
    model = tc.build_model(sigma, {"G12": (0, 1), "G21": (1, 0)})
    for kw in [dict(cap=0), dict(cap=-1), dict(rho=0.0), dict(delta=2.0)]:
        args = dict(n_environments=2, delta=1e-2, cap=100)
        args.update(kw)
        with pytest.raises(ValueError):
            tc.Certifier(model, **args)
    with pytest.raises(ValueError):
        tc.Certifier(model, n_environments=3, delta=1e-2)  # K > d
