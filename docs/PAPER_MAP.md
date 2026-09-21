# What in the package corresponds to what in the paper

The reference is the v21 manuscript, *Track-and-Certify: Joint Causal
Identification under Unknown Intervention Semantics*. Numbering below is that
version's: the main text carries Theorem 1, Propositions 1-3 and Corollary 1,
and the supplement numbers its own results independently.

## Implemented, and backed by a stated result

| Public entry point | Result it implements |
| --- | --- |
| `Certifier.observe`, the mixture e-process and its threshold | Theorem 1(ii), anytime certificate safety under any predictable sampling rule, including one that never terminates |
| `Certifier` with the default `sampling_policy="adaptive"` | Theorem 1(iii), constructive attainability: restarted scale-free entropic FTRL, forced exploration, cumulative C-tracking |
| `characteristic_time` — `tstar`, `w_star`, `active_kinds` | Theorem 1(i), the information lower bound, and its linear-program form in the supplement |
| `coupling_tax` | Proposition 1, the self-calibration decomposition. The returned `tax` is the total factor; see below |
| the exact target-assignment oracle in `_vendor/assignment_oracle.py` | Proposition 2, the exact assignment reduction that replaces factorial target enumeration |
| `solvable_instance` | the two-variable near-boundary family discussed under Proposition 1, extended with an independent third node and a second environment |
| `scalable_instance` | supplement Corollary 1, multiple marginal tasks |
| `build_model` refusing ray-identical declarations | the class-wide decision-distinctness requirement of the formal model |

Every public entry point appears in that table. Nothing the package exports
rests on a result the manuscript does not state.

## Removed in 0.2.0: the estimated-covariance path

Up to 0.1.1 the package exported `certified_envelope`, `robust_certifier`,
`estimate_covariance`, `RobustSetup`, `Envelope`, `Refusal` and a
`track-certify envelope` command, and described them as implementing a
"Theorem 4" and a "Corollary 2". The manuscript has neither, at v21 or
before. It assumes a known common covariance throughout, and its supplement,
"Estimated Covariance: Required Conditions", lists the four things an
anytime-valid extension would have to supply at once -- a spectral envelope
uniform over the candidate response subspaces, an amplitude envelope
controlling response-direction perturbation, a boundary that survives
propagating covariance uncertainty through the adaptive e-process, and an
allocation-game perturbation bound preserving positive separation -- and
states that none of them is established.

The construction was self-contained and its feasibility gate did refuse
before drawing interventional data, but the error level it ran at was not
covered by Theorem 1(ii), so it was removed rather than relabelled. The
threshold-inflation arguments it fed to `Certifier` (`penalty_dims`,
`beta_scale`, `beta_drift`) went with it; the vendored simulation driver
still accepts them, and its defaults reproduce the known-covariance rule.

## In the paper, not in the package

- Proposition 3, the geometry and stability of evidence-weighted planning, and
  the evidence-weighted allocator it describes. The shipped policies are
  `adaptive`, `uniform`, `oracle` and a plug-in `dtracking` comparator. The
  paper claims no optimality for the evidence-weighted allocator either.
- Corollary 1 of the main text, the active-value index, and the two forms in
  which it is usable. No entry point computes it.
- Every experimental panel. The package ships the method, not the evidence:
  no trajectories, no fitted results and no figures. Those live with the
  manuscript and its separate code artifacts.

## One naming difference worth knowing

`coupling_tax(...).tax` divides the joint characteristic time by the larger of
the two conditional times. In the paper's notation that is the total factor
`kappa_total = min(V_G, V_T) / V_J`, not the coupling factor
`kappa_coup = V_M / V_J`. The two coincide exactly when the allocation factor
`kappa_alloc` is one, which is the case in the near-boundary family that
`solvable_instance` builds and in the paper's own divergence example. Telling
them apart needs the marginal game value `V_M`, which this package does not
compute.
