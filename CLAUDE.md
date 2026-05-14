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
├── training/      Trajectory NPZ → LeRobot-style parquet (image + state + action)
├── planning/      Course YAML → Trajectory NPZ via cubic spline (MPC stub)
├── io/            YAML config loaders + build_frame_graph
└── cli/           smoke_test, visualize_frames, run_vla_episode, export_training_data,
                   visualize_waypoints, plan_trajectory
```

**Sensor decoupling.** Policies declare `required_modalities`; the orchestrator
builds a `SensorRig` from a registry (state, prompt, cameras, future
lidar/IMU…). A `CameraSensor` is one sensor type, not a global requirement.
Mock policies declare no modalities and run with just `StateSensor` — no
gsplat renders triggered.

## Skills index

`.claude/skills/README.md` is the authoritative index of composable
workflows in this repo (trajectory generation, training-data export,
orchestration, debugging). **Check it before writing a new script** —
almost every common task is already exposed as a skill that chains
with the rest.

## Producing training data

`falsify.training/` converts any `Trajectory` (NED positions + quaternions
+ times) into a LeRobot-style parquet (HuggingFace Image features, embedded
PNG bytes) that DroneVLA2.0's training pipeline ingests directly. The
contract is three swappable layers:

- **Trajectory** producers — live VLA rollout, replay from a `vla_io/`
  run dir, mock straight-line / helix, future FiGS-MPC and SplatNav.
- **Scene** — same scene YAMLs the rest of falsify uses.
- **Embodiment** (`configs/embodiments/*.yaml`) — declarative
  state/action layout + camera column mapping + channel order.

See `src/falsify/training/CLAUDE.md` for the contract, the parquet schema,
and how to add a new embodiment or trajectory producer. The skills under
`.claude/skills/falsify-*` chain the steps for higher-level workflows.

Status: **v0 feature-complete.** All ten subpackages exercise end-to-end
through the smoke CLI (mock policy → sensor rig → policy → perturbations →
detector → recovery → visualization). The MPC-backed integrator and concrete
Splat-MOVER environment perturbations are deferred until the full FiGS env
and the new gsplat asset land — both wired in behind documented seams.
Per-package `CLAUDE.md` files document the contracts of each module.

## External submodules

`external/` holds the three Stanford-MSL repos (FiGS, splatnav, Splat-MOVER)
as proper git submodules pinned at the SousVide-validated SHAs.

`data/` is a symlink to the active gsplat asset bundle (not tracked in git).
The v0 scenes live in `data/gate_scenes_export/` — two sagesplat exports
(`left_scene/`, `right_scene/`) sharing a JOINT mocap frame (`left_scene`'s
mocap by construction; `right_scene` aligned to it via ICP). Per-scene
nerfstudio transforms come from
``data/gate_scenes_export/objects_final/joint_mocap_to_nerf.json``, loaded by
the ``sim3_matrix_file`` transform loader (4×4 in JSON → Sim3). Scene YAMLs
in ``configs/scenes/`` declare the same canonical frame names (``mocap``,
``ned``, ``ns``, ``cam_*``) so all higher-level code is scene-agnostic.

## Setup

```bash
git submodule update --init --recursive   # (once we git-init this repo)
uv sync
source .venv/bin/activate
```

### Known env caveat — `gsplat` CUDA JIT

The active venv is symlinked to `~/code/SousVide/.venv` and pinned via
`~/code/SousVide/uv.lock` (torch 2.1.2+cu121, numpy 1.26.4,
gsplat 0.1.13). `gsplat`'s CUDA extension is built lazily via
`torch.utils.cpp_extension.load()` into
`~/.cache/torch_extensions/py311_cu121/gsplat_cuda/`.

**Root cause of the recurring rebuild failure.** CUDA 12.0's nvcc
(V12.0.140) officially supports gcc up to 12; Ubuntu 24.04 ships gcc 13.3
as `/usr/bin/gcc`. The combo of nvcc 12.0 + gcc 13 rejects the pybind11
template syntax in torch's headers with:

```
pybind11/cast.h:45: error: expected template-name before '<' token
```

Any time the JIT cache is invalidated (gsplat reinstall, torch upgrade,
`rm -rf` of the cache, kernel-update-induced build-key change, etc.), the
rebuild runs against whatever `/usr/bin/gcc` resolves to and fails. The
fix is to pin nvcc's host compiler to `gcc-11` (also installed on the
system) before any process that may JIT-build a torch extension:

```bash
source tools/env.sh   # exports CC=gcc-11, CXX=g++-11, NVCC_PREPEND_FLAGS=-ccbin /usr/bin/g++-11
```

The exports only matter at build time; sourcing again with a warm cache
is a no-op. `tools/env.sh` also fixes the SousVide-venv PYTHONPATH so the
falsify package and the external/{FiGS,splatnav} submodules import.

If a rebuild still fails after sourcing `tools/env.sh`:

1. Wipe the cache (`rm -rf ~/.cache/torch_extensions/py311_cu121/gsplat_cuda/`)
   so ninja rebuilds from scratch with the pinned toolchain. Touching
   individual file mtimes does NOT help — ninja keys on a build-command
   hash, not mtimes.
2. Surgical reinstall that leaves torch/numpy alone:
   `uv pip install --reinstall-package gsplat --no-deps gsplat==0.1.13`.
3. For mask / move correctness checks that don't need photorealistic
   renders, use `paint_gaussian_mask` or `inspect_scene_plotly` — they
   read `pipeline.model.means` directly and never enter `gsplat_cuda`.

**Do NOT run `uv pip install --force-reinstall gsplat`.** gsplat's deps
are unpinned, so a force-reinstall re-resolves against PyPI HEAD and
cascades the entire ML stack: torch 2.1.2+cu121 → 2.12.0+cu130,
numpy 1.26.4 → 2.4.4, nvidia CUDA libs 12.1 → 13.0. `cv2` (compiled
against numpy 1.x) then fails to import, taking down nerfstudio before
gsplat even gets reached. SousVide and falsify both share this venv so
the damage is doubled. Recovery if it happens anyway:
`cd ~/code/SousVide && uv sync` restores the lockfile state — but the
sync also removes ad-hoc-installed packages, so reinstall
`pyarrow` and `pytest` additively afterwards
(`uv pip install pyarrow pytest`).

Anything that needs the gsplat CUDA path (ns-viewer, `GSplatRenderer`,
`run_vla_episode`) depends on the cached `.so` staying valid. The
mask-authoring + applier code paths in `sim/scene_edits.py` are
CPU-only and unaffected either way.

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
