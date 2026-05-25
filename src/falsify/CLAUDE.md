# `src/falsify/` — package layout

| Subpackage | Purpose | Phase | Status |
|------------|---------|-------|--------|
| `training/` | `Trajectory` NPZ → LeRobot-style parquet via `TrainingDataExporter`; `EmbodimentSpec` decouples state/action layout + camera mapping | 7 | done |
| `planning/` | Course YAML → Trajectory NPZ. `plan_spline` (geometric) + `plan_mpc` (FiGS `VehicleRateMPC` w/ acados, SQP-RTI by default) shipped | 7 | spline + MPC done |
| `geometry/` | Frame-tagged types (`Point`, `Pose`, `Trajectory`, `PointCloud`) + `FrameGraph` + YAML-driven transform registry | 1 | done |
| `sim/`      | FiGS simulator wrapper + GSplat renderer + `DroneState` + runtime body↔world hinge + `scene_edits` (rigid AABB + duplicate-AABB) | 2 | done (v0; replay integrator only — closed-loop FiGS dynamics deferred) |
| `sensors/`  | Pluggable observation pipeline (`Sensor`, `SensorRig`, `StateSensor`, `PromptSensor`, `CameraSensor`, `build_sensor_rig` factory) | 2 / 3 | done |
| `policy/`   | `Policy` ABC, `Observation`, mock policies, `VLAPolicy` (legacy OpenPI client), `PiGatewayPolicy` (pi-inference-client gateway — Pi-hosted or self-hosted via `pi_local_bridge/`; optional async RTC mode) | 3 / 6 | done |
| `safety/`   | Pluggable `SafetyCriterion`s (bounds / velocity / tilt / OBB-vs-point-cloud collision / aperture miss-gate), `FailureDetector` (last-safe tracking, per-criterion reset), post-hoc directional-transit classifier | 4 | done |
| `recovery/` | Multi-backend recovery planners: `CoursedMpcPlanner` (default — FiGS MPC over a course YAML from `last_safe_state`), `SplatNavPlanner` (A*+spline, NED in/out), `SplatNavMpcPlanner` (A*-seeded MPC), plus `sample_recovery_seed` (failure-type-aware seed picker over `safe_history`) | 5 | done |
| `perturbations/` | Three surfaces (`PerturbationSuite`, action/observation/environment), JSON manifest, seedable RNG. Environment surface ships `GateRigidPerturbation` (rigid xyz+yaw jitter on gate Gaussians) | 6 | done (multi-object / opacity / color edits pending) |
| `cem/`      | Cross-Entropy Method falsification — `GaussianBoxDistribution` over the 6-d `(start_dxyz, gate_dxy, gate_dyaw)` vector + per-`FailureType` continuous cost functions in `scorer.py` + trial-card writer | 6+ | v0 (COLLISION_GATE primary) |
| `eval/`     | Shared evaluation-pipeline helpers — `sampling.py` derives deterministic per-trial RNGs from a campaign master seed | 8 | done |
| `orchestrator/`  | `run_episode` (with optional perturbations / recovery / detector factories, RNG passthrough, initial-state + perturbation override hooks for replay) | 3+ | episode done |
| `visualization/` | `dump_episode` (per-frame `.ply`s), `html_replay` (plotly), `eval_report` (per-campaign trajectory overlay + outcome stacked-bar HTML) | 4+ | done |
| `io/`            | YAML config loaders + `build_frame_graph` | 1+ | configs done |
| `cli/`           | `smoke_test`, `run_vla_episode`, `export_training_data`, `plan_trajectory`, `visualize_frames`, `visualize_waypoints`, `inspect_scene_plotly`, `preview_scene_nsviewer`, `author_gaussian_mask`, `paint_gaussian_mask`, `perturb_course`, `combine_lerobot` | 3+ | done |

**Cross-cutting rule:** nothing across module boundaries passes a raw 3-vector
or 4×4 matrix as `np.ndarray` for positions/poses. Use the geometry types
— they carry their frame tag and let `FrameGraph` convert safely.
