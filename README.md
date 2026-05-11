# falsify

Falsification framework and Gaussian-splat training dojo for multimodal drone
policies (VLAs). Ties three Stanford-MSL research components together with
explicit, config-driven coordinate frames:

- **[FiGS](https://github.com/StanfordMSL/FiGS)** — flight-in-Gaussian-splats simulator (ACADOS dynamics, gsplat rendering).
- **[SplatNav](https://github.com/chengine/splatnav)** — collision-free A\*+spline planning over Gaussian splats; powers recovery.
- **[Splat-MOVER](https://github.com/StanfordMSL/Splat-MOVER)** — gsplat scene editing; powers environment perturbations.

## Why a rewrite?

The previous prototype (`~/code/SousVide`) worked but became hard to extend
because coordinate transforms were spaghetti: raw `np.ndarray`s flowed across
modules with frame context only in variable names, transform matrices were
duplicated for slightly different purposes, and adding a new scene/gsplat
required code changes. **falsify enforces frame tags everywhere** —
positions/poses/trajectories/point clouds carry their frame, conversions go
through one config-driven `FrameGraph`, and "adding a new frame" means
editing a scene YAML.

## Quickstart

```bash
git clone <this repo>
cd falsify
git submodule update --init --recursive

uv sync                            # full env (FiGS, splatnav, splat-mover, CUDA stack)
source .venv/bin/activate

# Mock-policy smoke test (no GPU needed):
PYTHONPATH=src python -m falsify.cli.smoke_test \
    --config configs/falsification/smoke.yaml

# Trigger failure + stubbed recovery (still no GPU):
PYTHONPATH=src python -m falsify.cli.smoke_test \
    --config configs/falsification/smoke_recovery.yaml \
    --stub-recovery

# Run with action perturbations (also no GPU):
PYTHONPATH=src python -m falsify.cli.smoke_test \
    --config configs/falsification/smoke_with_perturbations.yaml \
    --stub-recovery

# Full pipeline against the real GSplat + real SplatNav (needs CUDA):
PYTHONPATH=src python -m falsify.cli.smoke_test \
    --config configs/falsification/smoke_recovery.yaml
```

Each run writes a timestamped directory under `runs/<stamp>/`:
- `episode_summary.json` — frame-tagged start/goal/end, failure record, recovery summary, perturbation manifest.
- `frames/*.ply` — one PLY per (entity × frame), each header tagged with the frame name. Open in meshlab / open3d / blender to debug alignment.
- `episode.html` — plotly replay in a chosen frame.

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

The geometry layer, sensors, mock policies, safety, recovery (stub), visualization,
perturbations, and a stubbed VLA round-trip all run with no GPU. ~90 tests.

## Frame contract

The single most important design rule: every position-like value carries its
`Frame`. All conversions go through a `FrameGraph` built from the scene
YAML. The canonical frames shipped with the default scene:

```
            FrameGraph for configs/scenes/left_gate.yaml
            =============================================

       (xyzw quaternions everywhere; right-handed frames)

                    ┌──────────────────┐
                    │   mocap          │  Z-up motion-capture world
                    │   (ground truth) │
                    └────────┬─────────┘
                             │  permutation "perm5"
                             │  (SE3: diag(1,-1,-1))
                  ┌──────────┴───────────┐
                  ▼                      ▼
            ┌──────────┐           ┌────────────┐
            │   ned    │           │   colmap   │  raw SfM frame
            │ (FiGS    │           │            │
            │  dynamics│           └─────┬──────┘
            │  + state)│                 │
            └──────────┘                 │  dataparser_transforms.json
                                         │  (Sim3 ≈ scale 0.16 + R + t)
                                         ▼
                                   ┌──────────┐
                                   │    ns    │  nerfstudio-internal
                                   │ (gsplat  │  (where SplatNav plans)
                                   │  lives)  │
                                   └──────────┘

  Runtime-only (the body's pose in NED is the drone state, applied in sim/):

            ┌──────────┐     fixed body→camera SE3      ┌──────────────┐
            │ cam_body │ ─────────────────────────────► │ cam_forward  │
            │ (IMU     │ ─────────────────────────────► │ cam_downward │
            │  origin) │                                └──────────────┘
            └──────────┘
                ▲
                │ runtime: T_body_to_ned built from DroneState.quat + .pos
                │
            ┌──────────┐
            │   ned    │
            └──────────┘
```

The mocap↔ned↔colmap↔ns edges are **static** transforms in the
`FrameGraph`. The body↔world edge is the **runtime** drone state, applied
inside `sim.poses.camera_to_world_pose` — see
`src/falsify/geometry/CLAUDE.md` for the static-vs-runtime split.

### Adding a new frame

Edit the scene YAML — no Python required:

```yaml
# configs/scenes/<my_scene>.yaml
frames:
  - { name: my_new_frame, notes: "what this is" }
transforms:
  - { src: my_new_frame, dst: ned, type: se3_inline,
      R: [[...]], t: [...] }
```

Re-run; `FrameGraph.convert(point, to="my_new_frame")` works immediately via
BFS through the registered edges. See
`src/falsify/geometry/CLAUDE.md` for the loader registry and the recipe for
adding a new transform *type* (e.g. an HDF5-based extrinsic).

## Architecture

| Package | Purpose |
|---------|---------|
| `geometry/` | Frame-tagged `Point` / `Pose` / `Trajectory` / `PointCloud`, `SE3` / `Sim3`, `FrameGraph`, YAML transform loaders |
| `sim/` | `Simulator` (trajectory-replay v0; FiGS-MPC integrator deferred), `GSplatRenderer` (lazy), `DroneState`, runtime body↔world hinge |
| `sensors/` | Pluggable observation pipeline — `Sensor` ABC, `SensorRig`, `StateSensor`, `PromptSensor`, `CameraSensor`, factory |
| `policy/` | `Policy` ABC, `Observation`, `MockStraightLine` / `MockNoisy`, `VLAPolicy` (OpenPI websocket, NED↔MOCAP boundary) |
| `safety/` | Pluggable safety criteria (bounds, velocity, tilt), `FailureDetector` with last-safe tracking |
| `recovery/` | `SplatNavPlanner` — NED in, NED out; lazy splatnav backend, stubbable for tests |
| `perturbations/` | Three surfaces (action / observation / environment), `PerturbationSuite`, JSON manifest, seedable RNG |
| `orchestrator/` | `EpisodeConfig`, `run_episode`, `FalsificationEpisode` |
| `visualization/` | Per-frame PLY dumps + plotly html replay |
| `io/` | YAML config loaders, `build_frame_graph` |
| `cli/` | `smoke_test` entry point |

Each package has its own `CLAUDE.md` with the detailed contract.

## What's deferred for v0

- **MPC-backed integrator.** The simulator follows the policy's emitted
  trajectory directly. Once acados + the full FiGS env is verified locally,
  swap in `VehicleRateMPC` inside `sim.simulator._step_replay`.
- **Concrete environment perturbations.** Splat-MOVER integration lands once
  a new gsplat asset arrives; the `StubEnvironmentPerturbation` raises so
  forgetting to remove it fails loudly.
- **Campaign sweep.** `run_campaign` (seed × perturbation sweeps) — straight
  loop over `run_episode`; will land alongside the first VLA campaign run.

## Co-developing with submodules

The three submodules under `external/` are pinned at the same SHAs SousVide
uses, so a checkout reproduces a known-good combination. To update one:

```bash
cd external/FiGS && git checkout <new-sha> && cd -
git add external/FiGS
```
