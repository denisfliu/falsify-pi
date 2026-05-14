---
name: falsify-perturb-course
description: Generate variant course YAMLs by perturbing one waypoint up / down / left / right of its nominal position. Use to produce corrective-maneuver demonstrations — the planner + exporter turn each variant into one episode of training data, teaching the policy "I'm off; recover".
---

# falsify-perturb-course

Nudges one named waypoint of a base course along a sampled direction, in
sampled magnitudes, and writes the result as N standalone course YAMLs.
Downstream pipeline (plan → render → parquet) is unchanged — each
variant flows through as if it were hand-authored.

## When to use

You have a working course (`through_left_gate.yaml`) that produces a
"standard" demonstration. You want the policy to also see corrective
maneuvers: the drone starts somewhere off the ideal path and has to
recover. Pick the waypoint where you want the perturbation to occur
(typically the approach) and let this skill enumerate the spread.

## Directions

- `center` — no perturbation (baseline / standard demo); always magnitude 0.
- `up` / `down` — ±z_mocap (world up / down).
- `left` / `right` — perpendicular to local flight direction, in the
  xy-plane. *Body-relative*: a positive "right" perturbation is the
  same physical side regardless of which gate the drone is approaching,
  because the direction is computed from the local heading (chord
  through the perturbed waypoint).

## Procedure

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.perturb_course \
    --course configs/courses/through_left_gate.yaml \
    --waypoint approach \
    --out configs/courses/through_left_gate_variants \
    --samples-per-mode 5 \
    --magnitude-range 0.2 0.5 \
    --seed 0
```

This writes 25 course YAMLs (5 modes × 5 samples), each preserving the
base course's other fields. The output filenames sort by mode then
sample index (`..._center_00.yaml`, `..._down_03.yaml`, ...).

### Visualize the spread before rendering

The plotly inspector accepts a directory of courses — every variant
shows up as a distinct color over the scene:

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.inspect_scene_plotly \
    --scene configs/scenes/left_gate.yaml \
    --courses-dir configs/courses/through_left_gate_variants \
    --out runs/inspect/left_gate_variants.html
```

Open the HTML and confirm the perturbed splines still thread the gate
and don't clip the table. If the magnitude range pushes splines into
solid objects, dial it down.

### Render the whole spread

After the spread looks right:

```bash
# 1) Plan each variant → trajectory NPZ.
mkdir -p runs/trajectories/left_gate_variants
for c in configs/courses/through_left_gate_variants/*.yaml; do
  name=$(basename "$c" .yaml)
  PYTHONPATH=src .venv/bin/python -m falsify.cli.plan_trajectory \
      --course "$c" --scene configs/scenes/left_gate.yaml \
      --out "runs/trajectories/left_gate_variants/${name}.npz" \
      --prompt "go through the gate and hover over the stuffed animal"
done

# 2) Bulk-export — one parquet per variant, indices auto-assigned.
PYTHONPATH=src .venv/bin/python -m falsify.cli.export_training_data \
    --trajectories-dir runs/trajectories/left_gate_variants \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/datasets/left_gate_variants \
    --episode-index 0
```

The `--trajectories-dir` mode reuses the gsplat across episodes — the
~30 s scene load is paid once, not per variant.

## Reproducibility

`--seed` controls magnitude sampling. Given the same base course +
seed + sampling args, the variant YAMLs are byte-identical.

## Hands off to

- **`falsify-trajectory-from-waypoints`** — plan each variant.
- **`falsify-export-parquet`** / **`falsify-orchestrate-batch`** —
  render and emit one parquet per variant.

## Gotchas

- The perturbation is applied to *the course definition*, not the
  spline. The cubic spline planner curves through the perturbed
  waypoint, so the entire trajectory bends — the drone doesn't snap
  back to the nominal path. This is the desired training signal:
  the policy must learn the curve back to the gate from any approach.
- Body-relative directions need a non-degenerate local heading. If the
  perturbed waypoint sits at a 180° fold in the course, "left" and
  "right" become ill-defined. Avoid by adding a separating waypoint or
  by using world-axis directions (`up`/`down`).
- Magnitudes are in MOCAP meters. Verify in the inspector before
  rendering — the gsplat extent is finite and large perturbations can
  push the drone outside the trained workspace.
