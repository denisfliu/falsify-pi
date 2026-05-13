# `src/falsify/` — package layout

| Subpackage | Purpose | Phase | Status |
|------------|---------|-------|--------|
| `training/` | `Trajectory` NPZ → LeRobot-style parquet via `TrainingDataExporter`; `EmbodimentSpec` decouples state/action layout + camera mapping | 7 | done |
| `geometry/` | Frame-tagged types (`Point`, `Pose`, `Trajectory`, `PointCloud`) + `FrameGraph` + YAML-driven transform registry | 1 | done |
| `sim/`      | FiGS simulator wrapper + GSplat renderer + `DroneState` + runtime body↔world hinge | 2 | done (v0; FiGS-MPC integrator deferred) |
| `sensors/`  | Pluggable observation pipeline (`Sensor`, `SensorRig`, `StateSensor`, `PromptSensor`, `CameraSensor`, `build_sensor_rig` factory) | 2 / 3 | done |
| `policy/`   | `Policy` ABC, `Observation`, mock policies, `VLAPolicy` (OpenPI client, NED↔MOCAP boundary) | 3 / 6 | done |
| `safety/`   | Pluggable `SafetyCriterion`s (bounds/velocity/tilt), `FailureDetector` (last-safe tracking) | 4 | done |
| `recovery/` | `SplatNavPlanner` — `plan(start_ned, goal_ned) → Trajectory["ned"]`, lazy backend | 5 | done |
| `perturbations/` | Three surfaces (`PerturbationSuite`, action/observation/environment), JSON manifest, seedable RNG | 6 | done (env stubs pending Splat-MOVER) |
| `orchestrator/`  | `run_episode` and (later) `run_campaign` | 3+ | episode done |
| `visualization/` | `dump_episode` (per-frame `.ply`s), `html_replay` (plotly) | 4+ | done |
| `io/`            | YAML config loaders, (later) episode parquet/json store | 1+ | configs done |
| `cli/`           | `smoke_test` runs end-to-end with `--stub-recovery` flag | 3+ | done |

**Cross-cutting rule:** nothing across module boundaries passes a raw 3-vector
or 4×4 matrix as `np.ndarray` for positions/poses. Use the geometry types
— they carry their frame tag and let `FrameGraph` convert safely.
