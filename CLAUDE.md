# CLAUDE.md — falsify

This file is the source of truth for module boundaries in **falsify**, a clean
falsification framework and GSplat training dojo for multimodal drone policies
(VLAs) over Gaussian Splatting environments. It must be kept current — if a
code change makes a section here stale, the change is incomplete.

## Project goal

Falsify ties three external research components into one coherent system:

1. **FiGS** (`external/FiGS/`) — flight-in-gaussian-splats simulator with
   ACADOS dynamics, gsplat rendering, and an MPC tracker.
2. **SplatNav** (`external/splatnav/`) — collision-free A* + spline planner
   over Gaussian splat scenes; used for failure recovery.
3. **Splat-MOVER** (`external/Splat-MOVER/`) — gsplat scene editing; used to
   perturb objects in the scene as environment-level falsifications.

A *VLA policy* observes drone state plus dual-camera renders from the gsplat
and emits a trajectory. The simulator rolls it out. A failure detector
monitors the rollout; on failure, the orchestrator extracts the last safe
state and asks SplatNav to plan a recovery trajectory to the goal.

## Why this is a rewrite

The original (`~/code/SousVide`) worked but became hard to extend because
coordinate transforms were spaghetti — raw `np.ndarray`s flowing across
modules with frame context only in variable names, and duplicated matrices for
slightly different purposes (`ned_to_ns_position` vs `ned_to_ns_render`).
This rewrite enforces:

- Every `Point`/`Pose`/`Trajectory`/`PointCloud` carries its **frame tag**.
- A single `FrameGraph` registry owns all transforms for the active scene.
- The set of frames + transforms is **data, not code** — declared in scene
  YAMLs. Adding a new frame or swapping a transform is a YAML edit.
- The drone has **two cameras** (forward + downward) declared in the
  drone-frame YAML; cameras are addressed by name everywhere.

## Architecture map

```
src/falsify/
├── geometry/      frame-tagged types + FrameGraph + YAML-driven build
├── sim/           FiGS wrapper + gsplat renderer + DroneState + body↔world hinge
├── sensors/       pluggable observation pipeline (Sensor, SensorRig, StateSensor, CameraSensor, …)
├── policy/        Policy ABC, Observation, mocks, VLAPolicy (OpenPI client)
├── safety/        Pluggable safety criteria + FailureDetector with last-safe tracking
├── recovery/      SplatNavPlanner — NED in, NED out
├── perturbations/ three surfaces (action / observation / environment) + JSON manifest
├── orchestrator/  run_episode + FalsificationEpisode
├── visualization/ per-frame ply dumps + html replay
├── io/            YAML config loaders + build_frame_graph
└── cli/           smoke_test (with --stub-recovery for env-less runs)
```

**Sensor decoupling.** Policies declare `required_modalities`; the orchestrator
builds a `SensorRig` from a registry (state, prompt, cameras, future
lidar/IMU…). A `CameraSensor` is one sensor type, not a global requirement.
Mock policies declare no modalities and run with just `StateSensor` — no
gsplat renders triggered.

Status: **v0 feature-complete.** All ten subpackages exercise end-to-end
through the smoke CLI (mock policy → sensor rig → policy → perturbations →
detector → recovery → visualization). The MPC-backed integrator and concrete
Splat-MOVER environment perturbations are deferred until the full FiGS env
and the new gsplat asset land — both wired in behind documented seams.
Per-package `CLAUDE.md` files document the contracts of each module.

## External submodules

`external/` holds the three Stanford-MSL repos (FiGS, splatnav, Splat-MOVER)
as proper git submodules pinned at the SousVide-validated SHAs. `data/`
is a symlink to SousVide's data for v0 (large binaries, not tracked in git);
when a new gsplat asset lands, `data/` will be repointed and a new scene
YAML will declare its frames and transforms.

## Setup

```bash
git submodule update --init --recursive   # (once we git-init this repo)
uv sync
source .venv/bin/activate
```

Environment variables (see SousVide CLAUDE.md):
- `LD_LIBRARY_PATH` includes `external/FiGS/acados/lib`
- `ACADOS_SOURCE_DIR=external/FiGS/acados/`
- `CUDA_HOME` → local CUDA 11.8

Run tests:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Run smoke test (once Phase 3 lands):
```bash
PYTHONPATH=src python -m falsify.cli.smoke_test --config configs/falsification/smoke.yaml
```

## Frame contract (lives in `src/falsify/geometry/CLAUDE.md`)

The canonical frames shipped with falsify are `ned`, `mocap`, `colmap`,
`ns`, `cam_body`, `cam_forward`, `cam_downward` — but the geometry layer
treats frames as data, so any scene YAML may declare more. See
`src/falsify/geometry/CLAUDE.md` for the full contract and a recipe for
adding new frames or transforms.
