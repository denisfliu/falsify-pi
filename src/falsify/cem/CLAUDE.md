# `falsify.cem/` — Cross-Entropy Method falsification

**Status:** v0. The optimize-then-exploit chain ships end-to-end; multi-
scene optimization runs sequentially (one CEM loop per scene), and
``COLLISION_GATE`` is the first-class objective. The other three failure
types (``COLLISION_OTHER``, ``MISS_GATE``, ``GOAL_NOT_REACHED``) share
the same framework with simpler proxy costs — refine if/when they
become a primary target.

## Two-phase workflow

1. **Optimize** — ``scripts/cem/run_cem_campaign.py`` refits a truncated
   multivariate Gaussian over the 6-d perturbation vector to drive up a
   continuous cost for the chosen `FailureType`. Emits one
   ``converged_distribution.json`` per scene under
   ``runs/cem_campaigns/<name>/<scene>/``.
2. **Exploit** — the eval-suite scenario YAML at
   ``configs/eval_suite/cem_*_exploit.yaml`` declares
   ``recipe.cem_distribution`` and lets the existing
   ``generate_eval_bundles.py`` sample trial cards from the JSON file.
   Downstream artifacts are byte-identical in shape to a normal eval
   campaign, so the A/B comparison against ``baseline_cem_box.yaml``
   falls out of ``summarize_eval_campaign.py`` for free.

## Parameter vector — 6-d, gate_dz pinned

```
θ = [start_dx, start_dy, start_dz, gate_dx, gate_dy, gate_dyaw]
```

Gate dz is **never** a free dimension — gates don't levitate above the
ground. The bounds vector's gate-z slot is enforced to 0 in
`distribution._box_to_bounds_vec`, and `GaussianBoxDistribution.unpack`
always emits 0 there in the trial-card output.

## Module layout

| File | Purpose |
|---|---|
| `distribution.py` | ``GaussianBoxDistribution`` — sample / refit / save_json / load_json. Truncation is implemented as element-wise clipping into ``[-bounds, +bounds]``. Initial cov = ``diag(bounds² / 3)`` (uniform-equivalent). |
| `scorer.py` | ``SceneContext.from_yamls`` pre-loads gate/other point clouds + aperture corners + goal once per scene. ``score_trial(rollout_npz, ctx, target, gate_deltas)`` returns a continuous cost. |
| `sampler.py` | ``write_cem_trial_cards`` turns N θs into trial-card JSON in the schema ``run_eval_campaign.py`` consumes. |
| `__init__.py` | Re-exports the constructor + the canonical param-name tuple. |

## Cost functions (in `scorer.py`)

Higher cost = more failure-like. CEM picks the top-K trials by cost.

| Target | Cost |
|---|---|
| `COLLISION_GATE` | ``−min_t signed_dist(drone_OBB(t), gate_cloud)`` |
| `COLLISION_OTHER` | same as above against the other cloud; ``−100`` penalty if a gate collision also happened (don't reward "table via gate") |
| `MISS_GATE` | per-plane-crossing offset beyond the aperture rectangle (mode-a only — modes b/c emerge incidentally as `MISS_GATE` rate when this cost is maximized) |
| `GOAL_NOT_REACHED` | distance from final position to goal in NED |

The OBB-to-pointcloud signed-distance helper
``obb_to_points_signed_distance`` is a standard box SDF (Quilez): negative
inside, positive outside, with sphere-cull around the body origin to
keep the per-step cost manageable on ~10k-point clouds.

## A/B test contract

The point of the framework is to **prove** that a CEM-converged
distribution produces more failures than uniform sampling over the same
box. Two scenario YAMLs lock that contract:

- ``configs/eval_suite/baseline_cem_box.yaml`` — uniform-within-box,
  ``master_seed: 7``. The baseline arm.
- ``configs/eval_suite/cem_collision_gate_exploit.yaml`` —
  ``recipe.cem_distribution.path`` points at the converged JSON, same N,
  same scenes, same ``master_seed``. The exploit arm.

The two scenarios share their box dimensions with
``configs/cem/collision_gate.yaml`` exactly. **If you widen one, widen
all three** — otherwise the A/B comparison is unfair.

Run both arms, then:

```bash
python scripts/eval/summarize_eval_campaign.py \
    runs/eval_campaigns/baseline_uniform_cem_box \
    runs/eval_campaigns/cem_collision_gate_exploit
```

A two-proportion z-test on the ``by_failure_type.COLLISION_GATE`` counts
finishes the comparison.

## Adding a new failure-type cost

1. Subclass nothing — just add a function ``cost_<name>(rollout, ctx,
   gate_deltas) -> dict`` returning at minimum ``{"cost": float}``.
2. Register it in ``COST_FUNCTIONS`` in ``scorer.py``.
3. Update the table above + ``configs/cem/<new>.yaml`` declaring
   ``target_failure_type:`` matching the new key.

## Gotchas

- **CEM iter-0 ≠ uniform recipe.** The Gaussian-with-uniform-variance
  prior matches the uniform mean/variance but is *not* sample-identical
  — the Gaussian draws can extend beyond the box and get clipped, which
  produces slightly biased samples near the boundary. The bias is small
  for the box widths we use (~10 cm); call this out if comparing iter-0
  CEM samples against a uniform run at the same seed.
- **Gate-collision penalty in ``COLLISION_OTHER``.** The −100 penalty
  assumes m-scale distance gaps. If you crank up the action-perturbation
  amplitude such that other-collisions are routinely > 1 m
  ("deep wrong-place"), the penalty may stop dominating and need
  re-tuning. Watch the elite-set composition when changing scenarios.
- **Per-scene independence.** Each scene runs its own CEM loop and gets
  its own ``converged_distribution.json``. If you want a single shared
  distribution across scenes (warm-start, transfer learning), the
  refactor is in ``run_cem_campaign._run_cem_for_scene`` — pull the
  sampling+scoring out into a per-scene call, refit once on the union
  of elites across scenes.
