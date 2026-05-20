"""`falsify.cem` — Cross-Entropy Method for falsification.

Two-phase usage:

1. **Optimize.** A CEM loop iteratively refits a truncated multivariate
   Gaussian over the 6-d perturbation vector
   ``[start_dx, start_dy, start_dz, gate_dx, gate_dy, gate_dyaw]`` against
   a continuous cost function for a chosen `FailureType`. The output of
   the loop is a single ``converged_distribution.json``.

2. **Exploit.** ``generate_eval_bundles.py`` loads that JSON and samples
   from it instead of from the uniform recipe. Downstream artifacts are
   byte-identical to a normal eval campaign — so the failure-rate of the
   converged distribution can be A/B-compared against uniform via the
   existing summarizer.

Module layout:

- ``distribution.py`` — ``GaussianBoxDistribution`` (sample / refit /
  save_json / load_json).
- ``scorer.py``       — continuous per-trial costs, one per supported
  `FailureType`, computed from each trial's ``rollout_states.npz``.
- ``sampler.py``      — wraps a distribution to write trial-card JSON
  with the same schema ``generate_eval_bundles.py`` emits.
- ``loop.py``         — the outer-loop driver invoked by
  ``scripts/run_cem_campaign.py``.

Gate dz is **never** sampled (gate cannot levitate); the parameter
vector is exactly 6-d and the bounds vector's gate-z entry is pinned
to 0.
"""

from falsify.cem.distribution import (
    GaussianBoxDistribution,
    PARAM_NAMES,
    PARAM_DIM,
)

__all__ = [
    "GaussianBoxDistribution",
    "PARAM_NAMES",
    "PARAM_DIM",
]
