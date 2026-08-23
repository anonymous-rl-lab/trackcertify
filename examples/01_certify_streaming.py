"""Streaming certification on a two-graph class (known covariance)."""
import numpy as np
import track_certify as tc
from track_and_certify_general import true_means  # vendored helper

sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
model = tc.build_model(sigma, {"X->Y": (0, 1), "Y->X": (1, 0)})

# Simulated lab: truth is X->Y with env targets (node0, node1), amplitudes (1.5, 1.2)
truth, amps = ("X->Y", (0, 1)), (1.5, 1.2)
rng = np.random.default_rng(7)
chol = np.linalg.cholesky(sigma)
means = true_means(model, truth, amps)

cert = tc.Certifier(model, n_environments=2, delta=0.01, cap=100_000)
while True:
    e = cert.next_environment()
    y = rng.normal(size=2) @ chol.T + means[e]      # your experiment here
    out = cert.observe(y)
    if out.stopped:
        break
print("certificate:", out.certificate, "| tau:", out.t, "| counts:", out.counts)
