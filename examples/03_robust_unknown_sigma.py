"""Estimated-covariance workflow: certify or refuse with explicit constants."""
import numpy as np
import track_certify as tc

rng = np.random.default_rng(0)
sigma_true = np.array([[1.0, 0.5], [0.5, 1.0]])
n0 = 600_000
obs = rng.multivariate_normal(np.zeros(2), sigma_true, size=n0)

try:
    setup = tc.robust_certifier(obs, {"X->Y": (0, 1), "Y->X": (1, 0)},
                                n_environments=2, delta=0.01, c_max=3.0)
except tc.Refusal as r:
    print("pre-start refusal:", r)
else:
    env = setup.envelope
    print(f"feasible: eps={env.eps:.4f}, eta_bar={env.eta_bar:.4f}, "
          f"beta_scale={env.beta_scale:.3f}, drift={env.beta_drift:.2e}/step")
    print("certifier ready:", type(setup.certifier).__name__)
