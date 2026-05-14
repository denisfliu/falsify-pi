---
name: falsify-author-gaussian-mask
description: Interactively tune the include / exclude AABBs of a `rigid_transform_aabb` scene-edit. Loads the gsplat once, classifies every Gaussian mean by mask membership, and renders an HTML you can iterate against — without re-running the renderer between AABB tweaks.
---

# falsify-author-gaussian-mask

The diagnostic complement to `falsify-debug-render` for scene edits.
When a moved-object edit leaves stragglers (e.g. the gate's "L-foot"
remaining at the old location after a translate), this tool tells you
*exactly* which Gaussians are stranded and why — so you can author a
tighter mask before paying the cost of a full ns-viewer reload.

## When to use

- After a `rigid_transform_aabb` edit, you see leftover Gaussians at
  the source location in ns-viewer.
- You want to know whether the leftovers are *inside* an exclude AABB
  (stranded) or *outside* the include AABB (uncaught) — the fix is
  different for each.
- You're authoring a new scene edit and want to spot-check the AABBs
  before launching the gsplat viewer.

## Inputs

- `--scene <scene.yaml>` — the scene whose `scene_edits` you want to
  tune. The CLI reads the first edit (or pick one with `--edit-name`).
- `--add-include "[mn]:[mx]"` / `--add-exclude "[mn]:[mx]"` — repeatable
  candidate AABBs added on top of whatever's in the YAML, so you can
  experiment without editing the scene file.
- `--neighborhood "[mn]:[mx]"` — crop visualization to a region of
  MOCAP. Default: union of all AABBs plus a 0.4 m buffer.
- `--max-points` (default 40000) — subsample after the neighborhood crop.

## Procedure

### 1. Baseline render: see what's stranded

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.author_gaussian_mask \
    --scene configs/scenes/center_gate.yaml \
    --out runs/inspect/mask_center_gate.html
```

Open the HTML. The legend gives counts; the canonical reading
(mirrors the applier — see `RigidTransformAABB`):

  ```
  broad   = target_aabb  \ (exclude_aabbs ∪ oriented_exclude_aabbs)
  precise = include_aabbs ∪ oriented_include_aabbs
  move    = broad ∪ precise
  ```

- **cyan** — Gaussians the edit WILL move (in `move`).
- **red** — inside `target_aabb` AND an exclude, **and not rescued by a
  precise include**. These are the stranded ones causing residual
  artefacts at the old location.
- **orange** — inside an exclude only (correctly preserved, e.g. table).
- **gray** — outside everything (hidden by default).

**Precise includes override excludes.** If you hand-paint an
`include_aabbs` (or `oriented_include_aabbs`) box, every Gaussian inside
it moves — even if it also falls inside an `exclude_aabbs` region. The
intent is that hand-curated boxes are ground truth.

If the red count is non-trivial, you've found the L-foot.

### 2. Iterate

Two paths, depending on what the red points represent:

#### 2a. Stranded points are residual gate Gaussians inside the table exclusion
The exclude is too greedy. Shrink it on the command line until the red
cluster shrinks:

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.author_gaussian_mask \
    --scene configs/scenes/center_gate.yaml \
    --add-exclude "[1.25, -0.85, 0.02]:[1.85, 0.40, 0.62]"  \
    --out runs/inspect/mask_center_gate.html
```

(Candidate excludes are tagged `candidate_exclude_N` in the legend.)

#### 2b. Stragglers are outside the include AABB entirely
Widen the include until they're picked up:

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.author_gaussian_mask \
    --scene configs/scenes/center_gate.yaml \
    --add-include "[0.30, 0.05, 0.0]:[1.40, 1.30, 2.10]" \
    --out runs/inspect/mask_center_gate.html
```

### 3. Commit

When the mask looks right, paste the AABBs you converged on into the
scene YAML's `scene_edits` block. Drop the `--add-*` flags; re-run the
authoring tool once more to confirm the YAML matches what you saw.
Then verify in the live viewer with `falsify-debug-render` /
`preview_scene_nsviewer`.

## Paint-brush tool — interactively discover missing regions

The static `author_gaussian_mask` CLI shows what's already captured. When
you need to *discover* an uncaught cluster (e.g., the L-foot of a gate),
use the Dash app `falsify.cli.paint_gaussian_mask`:

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.paint_gaussian_mask \
    --scene configs/scenes/center_gate.yaml \
    --port 8050
```

Open `http://localhost:8050`. Workflow:

1. **Set x, y, z range sliders** to carve out a candidate AABB in
   MOCAP. The yellow wireframe in the 3D view tracks it live; the
   counter shows "N Gaussians in box (M unpainted)".
2. Click **"Add box → painted"** to fill the box. Painted Gaussians
   go magenta in the 3D view.
3. Re-tune the sliders and click **Add** again to extend; the painted
   set is the *union* of every box you've added. Use **"Subtract box
   from painted"** to remove a region from the painted set.
4. **Undo** rolls back one button press; **Clear painted** resets.
5. The pre tag at the bottom updates live with the MOCAP AABB of the
   entire painted set, in paste-ready YAML form.
6. Copy that AABB into the scene's `scene_edits` — either as a new
   `include_aabbs` entry (once we support multi-AABB) or widen the
   existing `target_aabb_*` to encompass it.

### Why box-define rather than click or lasso?
Plotly Scatter3d clicks are unreliable across renderer/browser combos
(silent failure, no error). Range sliders give pixel-precise control
and the candidate-box overlay shows exactly what will be added before
you commit. For finding the L-foot of a gate, the workflow is usually
2-3 box-adds to cover the cluster.

When done, re-run the static `author_gaussian_mask` (or
`preview_scene_nsviewer`) to verify the committed AABB does the right
thing.

## Hands off to

- **`falsify-debug-render`** — the live-gsplat playbook for cross-
  checking the authored mask in nerfstudio's viewer.
- **`preview_scene_nsviewer`** — apply the committed `scene_edits` and
  view the actual rendered result.

## Gotchas

- **The gsplat load is the expensive step (~10 s)**. Re-running for each
  AABB tweak is fine; iterating dozens of times is slow. If you're
  exploring rapidly, keep the same neighborhood + max-points so plotly
  HTML cache hits are predictable.
- **Subsampling can hide rare stragglers.** If the red count is
  surprisingly low but you can see residue in the viewer, bump
  `--max-points` or crop with `--neighborhood` to keep more density in
  the suspect region.
- **The tool's "moved" classification is geometric — it doesn't apply
  the actual edit's transform.** The point of the tool is to validate
  the MASK, not the transform. To validate the transform, use the
  plotly inspector's `scene_objects` overlay or the live viewer.
- **MOCAP is the working frame here.** Even if the gsplat's native
  frame is NS, all coordinates and AABBs in the inspector are MOCAP for
  consistency with the rest of falsify's authoring workflow.
