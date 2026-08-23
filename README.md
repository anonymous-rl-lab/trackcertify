# track-certify

**Anytime joint certification of causal graphs and unknown intervention
targets** — a certify-or-refuse layer for sequential causal experiments,
with instance-optimal adaptive sampling.

```bash
pip install track-certify
```

Pure Python; depends only on `numpy` and `scipy`.

## The problem it solves

You run interventional experiments (gene perturbations, drug assays, A/B
interventions) where **the perturbed variables themselves are uncertain**: a
reagent may act off-target, weakly, or not at all. A causal conclusion is then
a *joint* claim — the graph **and** each environment's intervention target —
and every additional batch of samples costs money. Two questions matter more
than any point estimate:

1. **Where should the next observation be taken?**
2. **When is it safe to stop and certify the joint answer at error level δ?**

`track-certify` answers both with guarantees: an anytime-valid e-process
stopping rule (no peeking penalty, no multiplicity correction), a no-regret
allocation learner that provably attains the instance's information-theoretic
sample requirement `T*`, and an exact assignment oracle that replaces
factorial target enumeration with polynomial rectangular assignment. When it
cannot certify — budget cap reached, or model preconditions infeasible — it
**refuses rather than guesses**.

## 60 seconds: certify a joint answer

```python
import numpy as np
import track_certify as tc

# Declare the class: 2 candidate graphs over a known covariance,
# 2 environments with unknown, distinct targets of unknown amplitude.
sigma = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
model = tc.build_model(sigma, {"X->Y": (0, 1), "Y->X": (1, 0)})

cert = tc.Certifier(model, n_environments=2, delta=0.01, cap=200_000)

while True:
    e = cert.next_environment()      # which environment to sample now
    y = run_my_experiment(e)         # one fresh d-vector from environment e
    out = cert.observe(y)
    if out.stopped:
        break

out.certificate   # ("X->Y", (0, 1))  — graph + per-environment targets,
                  # wrong with probability <= 0.01 under optional stopping;
                  # None if out.refused (cap reached: refusal, not a guess)
```

The certifier never receives the true hypothesis, the amplitudes, `T*`, or
the optimal allocation.

## Before you sample: how hard is your instance?

```python
rep = tc.characteristic_time(model, ("X->Y", (0, 1)), amplitudes=(1.5, 1.2))
rep.tstar          # first-order samples per unit log(1/delta)
rep.w_star         # optimal budget split across environments
rep.active_kinds   # which wrong answers dominate: 'graph' / 'target' / 'coupled'
rep.budget(0.01)   # first-order sample estimate at delta = 0.01

tax = tc.coupling_tax(model, ("X->Y", (0, 1)), (1.5, 1.2))
tax.tax            # how much harder JOINT certification is than the harder
                   # conditional problem — near the identifiability boundary
                   # this diverges, and "estimate targets, then certify the
                   # graph" becomes unboundedly suboptimal
```

## Unknown covariance: certify or refuse, with explicit constants

```python
obs = collect_observational_samples()          # (n0, d), no interventions
try:
    setup = tc.robust_certifier(
        obs, {"X->Y": (0, 1), "Y->X": (1, 0)},
        n_environments=2, delta=0.01, c_max=3.0)
except tc.Refusal as r:
    print("infeasible for this n0/delta/d/c_max:", r)   # pre-start refusal
else:
    cert = setup.certifier    # threshold already carries the certified
                              # envelope inflation — nothing calibrated
```

## Benchmark instances with theorem-known difficulty

```python
inst = tc.solvable_instance(r=0.9)      # closed-form T*, coupled alternative
                                        # provably active (paper Prop. 3)
inst = tc.scalable_instance(rs=[0.8, 0.9], cs=[1.0, 1.2])
                                        # additive T*, 2^m graphs (Cor. 1)
res = tc.simulate(inst.model, inst.truth, inst.amplitudes, delta=1e-3,
                  rng=np.random.default_rng(0))
res.tau, res.correct                    # compare against inst.tstar_closed_form
```

Run `track-certify demo` for a self-contained certification demo, or
`track-certify envelope --n0 500000 --delta 0.01 --d 3 --c-max 1.0` to check
robust-mode feasibility from the command line.

## Guarantees, precisely

* **Validity.** With a correctly declared class and known covariance, the
  returned joint answer is wrong with probability at most `delta`, under
  arbitrary adaptive sampling and optional stopping (Gaussian mixture
  e-process + Ville's inequality; no union bound over hypotheses).
* **Optimality.** Expected stopping time satisfies
  `E[tau]/log(1/delta) -> T*` as `delta -> 0` — the change-of-measure lower
  bound for *any* delta-correct procedure, attained.
* **Estimated covariance.** With the split-sample envelope, validity is exact
  whenever the envelope's computable feasibility conditions hold; otherwise
  the workflow refuses before starting. Efficiency degrades by explicit,
  measured constant factors.
* **Refusal semantics.** A reached cap or an infeasible envelope produces a
  refusal, never a certificate. There is no procedure here that converts
  arbitrary data into a certificate.

**Model scope** (checked where checkable, refused when violated): finite
declared graph class sharing one positive-definite covariance; mean-shift
interventions with one unknown, distinct target per environment; Gaussian
noise; no hidden confounding. Duplicate or ray-identical graph declarations
are refused by default (they make hypotheses mutually uncertifiable).

## Relation to the paper

This package accompanies *Track-and-Certify: Anytime Joint Certification of
Causal Graphs and Unknown Intervention Targets* (under review) and implements
its method as stated. `tests/test_equivalence.py` proves trajectory
equivalence (identical actions, decisions, stopping times, and counts)
between the streaming `Certifier` and the underlying simulation driver over
shared noise tapes, on five instances under two allocation policies.

## License

MIT.
