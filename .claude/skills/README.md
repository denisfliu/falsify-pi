# falsify skills — index and usage

Before writing new code or scripts in this repo, **check this index first**.
Almost every common task in falsify is already exposed as a composable skill
in `.claude/skills/`. The skills speak a single canonical type
(`Trajectory` NPZ) so they chain cleanly.

## The skill landscape

```
            +----------------------------------------+
            |        Trajectory NPZ (canonical)      |
            |  times, positions_ned, quats_xyzw, …   |
            +----------------------------------------+
                   ▲                              ▲
   produced by                              consumed by
        │                                          │
 ┌──────┴──────────────────────┐         ┌─────────┴──────────┐
 │  trajectory-from-vla        │         │  export-parquet     │  ──►  one
 │  trajectory-from-replay     │         │  (renders + emits   │      LeRobot-
 │  trajectory-from-mock       │         │   training parquet) │      style
 │  trajectory-from-mpc   STUB │         └─────────┬───────────┘      parquet
 │  trajectory-from-splatnav S │                   │
 │  falsify-trajectory    STUB │                   ▼
 │  (post-hoc perturb)         │         ┌─────────────────────┐
 └─────────────────────────────┘         │  orchestrate-batch  │  ──►  many
                                         │  (loops the above)  │      parquets
                                         └─────────────────────┘

   Out-of-band: debug-render — invoke when anything looks wrong.
```

## When to use which skill — decision table

| You want to … | Skill | Status |
|---|---|---|
| Run the VLA against a scene, save a trajectory | `falsify-trajectory-from-vla` | done |
| Re-derive the trajectory of a previous VLA run (no re-inference) | `falsify-trajectory-from-replay` | done |
| Author a straight line / helix / scripted trajectory | `falsify-trajectory-from-mock` | done |
| Plan a feasible trajectory via FiGS-MPC | `falsify-trajectory-from-mpc` | **stub** |
| Plan a collision-free trajectory via SplatNav | `falsify-trajectory-from-splatnav` | **stub** |
| Inject perturbations / failures on an existing trajectory | `falsify-falsify-trajectory` | **stub** |
| Turn a trajectory + scene into one training parquet | `falsify-export-parquet` | done |
| Generate many parquets across scenes / sources | `falsify-orchestrate-batch` | done |
| Gray renders, wrong-direction camera, missing gaussians, JIT failure | `falsify-debug-render` | done |

**Stub** = the skill describes the intended interface but the underlying
code isn't there yet. The body of each stub points to the TODO list.

## Two canonical end-to-end recipes

### A. "Run a VLA episode, get a parquet"

```
falsify-trajectory-from-vla        # rolls out, writes runs/vla_<stamp>/
            │
            ▼  (the skill writes runs/vla_<stamp>/trajectory.npz)
falsify-export-parquet             # renders + emits the parquet
```

### B. "Bulk-generate training data across both gate scenes"

```
many invocations of any
trajectory-from-* skill            # produces many .npz files per scene
            │
            ▼
falsify-orchestrate-batch          # loops one renderer per scene, all NPZs
```

## Hard rules that already hold (don't violate)

1. **Frame conventions are pinned.** `R_mocap_from_ned = diag(1, -1, -1)`
   (SousVide perm5). FiGS' `Tw2g` is overridden from the active
   `FrameGraph` inside `GSplatRenderer` whenever `frame_graph=` is passed
   — never hand-author a different `Tw2g`. If you need a new frame or a
   new transform source, follow `src/falsify/geometry/CLAUDE.md`.

2. **Trajectory is always NED.** Positions and quaternions live in NED at
   the canonical-type boundary. The exporter converts to MOCAP per the
   embodiment's state-layout spec — that's the *only* place the NED → MOCAP
   conversion happens during export.

3. **Yaw quaternions are built in NED, not MOCAP.** Routing a
   MOCAP-frame `Trajectory` carrying quaternions through
   `frame_graph.convert(to="ned")` applies perm5 to the quats, which is
   a 180° flip about x — not the yaw remapping you want. The fix lives
   in `VLAPolicy.observe`; don't undo it.

4. **Channel order is configured per embodiment.** `carl_dual_mocap`
   emits BGR PNGs to match DroneVLA2.0's cv2-collected training data.
   If you author a new embodiment for a different consumer, set
   `channel_order` deliberately.

5. **The gsplat load is expensive (~30 s).** For more than one episode
   per scene, use `--trajectories-dir` or the Python orchestration
   pattern in `falsify-orchestrate-batch`. Never reload the gsplat per
   episode.

6. **gcc-11 for gsplat CUDA JIT.** Until the cached `.so` is warm,
   prefix calls with `CC=gcc-11 CXX=g++-11`. Ubuntu 24.04's gcc-13
   chokes on torch's pybind11 headers.

7. **Don't bypass `falsify-debug-render`.** If renders look wrong, run
   the playbook before guessing. The lessons captured there cost real
   debugging hours.

## Adding a new skill

If you find yourself doing something that doesn't fit any skill, add
one rather than writing a one-off script:

```
.claude/skills/<skill-name>/SKILL.md
```

with this frontmatter:

```yaml
---
name: <skill-name>
description: One-line "when to use this skill" — read by the harness when picking skills.
---
```

The body should: state inputs, give the canonical command(s), list
what's produced, name the skills it hands off to. Keep it tight —
skills are reference cards, not tutorials. Anchor everything in the
existing canonical types (`Trajectory` NPZ, scene YAML, embodiment YAML)
so it chains with the rest.

## Adding a new embodiment

A new VLA / training pipeline with a different schema is **a YAML edit,
not a code change**:

1. Copy `configs/embodiments/carl_dual_mocap.yaml`.
2. Edit `cameras` / `state` / `actions` lists. New state-field *names*
   (e.g., `roll_mocap`) need a getter in `exporter._STATE_GETTERS`;
   new action fields following the `d_<state_field>` pattern work out
   of the box.
3. Pass `--embodiment <new>.yaml` to `falsify-export-parquet`.

See `src/falsify/training/CLAUDE.md` for the full contract.

## Adding a new scene

Also a YAML edit:

1. Author `configs/scenes/<scene>.yaml` (see `left_gate.yaml` as the
   template). Declare frames + transforms via the standard loader
   types — `sim3_matrix_file` is the relevant one for new
   `joint_mocap_to_nerf.json` outputs.
2. Verify with `falsify-debug-render` step 6 (the round-trip numerical
   check) before plugging it into the pipeline.

## Reference docs

- `src/falsify/CLAUDE.md` — top-level package map.
- `src/falsify/training/CLAUDE.md` — training-data export contract.
- `src/falsify/geometry/CLAUDE.md` — frame contract.
- Memory `~/.claude/projects/-home-dfliu-code-falsify/memory/` — pinned
  conventions and incident notes from prior sessions.
