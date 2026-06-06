"""High-level training-data export pipeline.

Inputs (per episode):
- ``Trajectory`` (canonical NPZ-shaped value): the source-agnostic
  per-step pose sequence in NED.
- scene YAML + drone-frame YAML: define the `FrameGraph`, cameras,
  intrinsics. Same files the rest of falsify uses.
- ``EmbodimentSpec``: decides what features the parquet emits.
- A renderer callable matching ``GSplatRenderer.render(pose, intrinsics)``.

Outputs (per episode):
- ``<out>/episode_<id>/episode_<id>.parquet`` — LeRobot-style schema
  matching ``~/Downloads/episode_000008.parquet``.
- ``<out>/episode_<id>/manifest.json`` — provenance + config used.

This component is importable so orchestrators producing dozens or hundreds
of episodes can construct the (expensive) ``GSplatRenderer`` once and call
``export_episode`` repeatedly. See ``cli/export_training_data.py`` for the
shell entry point.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from falsify.geometry import FrameGraph, Frame, Point, Pose, quat_to_yaw_xyzw
from falsify.policy.camera_postprocess import CameraPostprocess
from falsify.policy.state_assembly import build_state_vector
from falsify.sensors.camera import CameraSpec, make_camera_sensor_from_yaml
from falsify.sim.dynamics_state import DroneState
from falsify.sim.poses import camera_to_world_pose

# Legacy alias for back-compat with anything importing from here.
_quat_to_yaw_xyzw = quat_to_yaw_xyzw

from .embodiment import CameraSpec as _EmbCameraSpec
from .embodiment import EmbodimentSpec
from .parquet_writer import ParquetWriter, SampleRow
from .trajectory import Trajectory, resample


RendererFn = Callable[[Pose, dict], tuple[np.ndarray, Optional[np.ndarray]]]


@dataclass
class ExportResult:
    parquet_path: Path
    manifest_path: Path
    n_frames: int
    duration_s: float
    elapsed_s: float


class TrainingDataExporter:
    """Render + emit one episode worth of training samples.

    Construct once per scene/embodiment; call ``export_episode`` per
    trajectory. The renderer instance is reused — gsplat loading is
    expensive, so amortise across episodes.
    """

    def __init__(
        self,
        *,
        scene_cfg: dict,
        frame_cfg: dict,
        frame_graph: FrameGraph,
        renderer: RendererFn,
        embodiment: EmbodimentSpec,
    ) -> None:
        self.scene_cfg = scene_cfg
        self.frame_cfg = frame_cfg
        self.frame_graph = frame_graph
        self.renderer = renderer
        self.embodiment = embodiment

        # Resolve embodiment cameras to falsify CameraSpecs once.
        self._camera_specs: dict[str, CameraSpec] = {}
        # Per-camera-column postprocess pipeline (resize → channel swap →
        # optional overlay). Built once at init; the only runtime call is
        # `pp.apply(rgb_native)` on the hot path.
        self._postprocess: dict[str, CameraPostprocess] = {}
        for cam in embodiment.cameras:
            if cam.source == "render":
                if cam.camera_name is None:
                    raise ValueError(
                        f"camera column {cam.column!r} declares source=render "
                        "but no camera_name"
                    )
                cam_yaml = frame_cfg["cameras"][cam.camera_name]
                sensor = make_camera_sensor_from_yaml(
                    cam.camera_name, cam_yaml, frame_graph,
                    renderer=renderer, body_to_world=camera_to_world_pose,
                )
                self._camera_specs[cam.column] = sensor.spec
                self._postprocess[cam.column] = CameraPostprocess.from_paths(
                    image_size=cam.image_size,
                    channel_order=cam.channel_order,
                    overlay_path=cam.gripper_overlay_path,
                )

        # Pre-resize static / zero images.
        self._cached_images: dict[str, np.ndarray] = {}
        for cam in embodiment.cameras:
            if cam.source == "zeros":
                self._cached_images[cam.column] = np.zeros(
                    (cam.image_size, cam.image_size, 3), dtype=np.uint8,
                )
            elif cam.source == "static":
                if not cam.static_path:
                    raise ValueError(
                        f"camera column {cam.column!r} source=static requires static_path"
                    )
                from PIL import Image as _Image
                img = np.asarray(_Image.open(cam.static_path).convert("RGB"))
                # Run the static image through the same postprocess as a
                # live render so channel order + overlay are consistent.
                pp = CameraPostprocess.from_paths(
                    image_size=cam.image_size,
                    channel_order=cam.channel_order,
                    overlay_path=cam.gripper_overlay_path,
                )
                self._cached_images[cam.column] = pp.apply(img)

    # ---- public API ----------------------------------------------------

    def export_episode(
        self,
        traj: Trajectory,
        out_dir: Path,
        *,
        episode_index: int = 0,
        index_offset: int = 0,
        task_index: int = 0,
        prompt_override: Optional[str] = None,
        hz_override: Optional[float] = None,
    ) -> ExportResult:
        """Render `traj` and emit a parquet under ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        target_hz = float(hz_override) if hz_override is not None else float(self.embodiment.fps)
        if len(traj) >= 2 and not self._matches_hz(traj, target_hz):
            traj = resample(traj, target_hz)

        prompt = prompt_override if prompt_override is not None else traj.prompt

        image_columns = tuple(c.column for c in self.embodiment.cameras)
        writer = ParquetWriter(
            image_columns=image_columns,
            state_dim=self.embodiment.state_dim(),
            action_dim=self.embodiment.action_dim(),
            episode_index=episode_index,
            task_index=task_index,
            index_offset=index_offset,
        )

        # Precompute per-frame states (in MOCAP, since the embodiment
        # currently emits MOCAP-frame fields).
        positions_mocap = self._ned_to_mocap_positions(traj.positions_ned)
        yaws_ned = np.array(
            [_quat_to_yaw_xyzw(q) for q in traj.quaternions_xyzw], dtype=np.float64,
        )
        yaws_mocap = -yaws_ned  # perm5 sign-flip on yaw

        # Build state vectors.
        states = np.zeros((len(traj), self.embodiment.state_dim()), dtype=np.float32)
        for i in range(len(traj)):
            states[i] = self._build_state(
                positions_mocap[i], yaws_mocap[i], positions_ned=traj.positions_ned[i],
                yaw_ned=yaws_ned[i],
            )

        # Build action vectors (delta to next state; first row = zeros per
        # DroneVLA2.0 convention).
        actions = np.zeros_like(states)
        if self.embodiment.first_action == "forward" and len(traj) >= 2:
            # First action = state[1] - state[0]; same rule applied below
            # for the rest. Useful for purely-future-deltas datasets.
            actions[0] = self._action_delta(states[0], states[1])
        for i in range(1, len(traj)):
            actions[i] = self._action_delta(states[i - 1], states[i])

        # Render each frame's image columns.
        for i in range(len(traj)):
            ds = self._state_to_drone(traj, i)
            images: dict[str, bytes] = {}
            for cam in self.embodiment.cameras:
                if cam.source == "render":
                    spec = self._camera_specs[cam.column]
                    pose = camera_to_world_pose(ds, spec.body_from_camera)
                    rgb, _depth = self.renderer(pose, spec.intrinsics)
                    img = self._postprocess[cam.column].apply(rgb)
                    images[cam.column] = self._encode_png(img)
                else:
                    images[cam.column] = self._encode_png(self._cached_images[cam.column])

            t_episode = float(traj.times[i] - traj.times[0])
            writer.add(SampleRow(
                images=images,
                state=states[i],
                actions=actions[i],
                timestamp_s=t_episode,
                frame_index=i,
            ))

        episode_id = f"{episode_index:06d}"
        parquet_path = out_dir / f"episode_{episode_id}.parquet"
        writer.flush(parquet_path)

        manifest = {
            "scene_key": self.scene_cfg.get("scene_key"),
            "embodiment": self.embodiment.name,
            "fps": target_hz,
            "prompt": prompt,
            "source": traj.source,
            "n_frames": len(traj),
            "duration_s": traj.duration_s,
            "episode_index": episode_index,
            "index_offset": index_offset,
            "task_index": task_index,
            "parquet": parquet_path.name,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return ExportResult(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            n_frames=len(traj),
            duration_s=traj.duration_s,
            elapsed_s=time.time() - t_start,
        )

    # ---- internals -----------------------------------------------------

    @staticmethod
    def _matches_hz(traj: Trajectory, hz: float, tol: float = 1e-3) -> bool:
        if len(traj) < 2:
            return True
        dt = float(np.diff(traj.times).mean())
        return abs(dt - 1.0 / hz) < tol

    def _ned_to_mocap_positions(self, pos_ned: np.ndarray) -> np.ndarray:
        """Bulk-apply ned→mocap via the FrameGraph (single edge for static frames)."""
        T = self.frame_graph.transform("ned", "mocap")
        R = T.R if not hasattr(T, "s") else T.s * T.R  # safe for SE3 or Sim3
        t = T.t
        return (R @ pos_ned.T).T + t

    def _build_state(
        self,
        pos_mocap: np.ndarray,
        yaw_mocap: float,
        *,
        positions_ned: np.ndarray,
        yaw_ned: float,
    ) -> np.ndarray:
        return build_state_vector(
            self.embodiment,
            pos_mocap=pos_mocap, yaw_mocap=yaw_mocap,
            pos_ned=positions_ned, yaw_ned=yaw_ned,
        )

    def _action_delta(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        # Compute deltas in the same indexed layout as ``state``. For yaw
        # entries (any state field whose name starts with "yaw"), the
        # difference is wrapped to [-π, π] (or not, per embodiment).
        delta = (curr - prev).astype(np.float32)
        for i, sfield in enumerate(self.embodiment.state):
            if sfield.name.startswith("yaw") and self.embodiment.yaw_wrap == "pi":
                d = (delta[i] + np.pi) % (2 * np.pi) - np.pi
                delta[i] = d
        # Pad slots (state name == "zero") always have delta 0; preserve that.
        for i, sfield in enumerate(self.embodiment.state):
            if sfield.name == "zero":
                delta[i] = 0.0
        # Map state-delta to action-layout: indices line up by position
        # because every action field is the delta of the matching state
        # field, or "zero". If a future embodiment diverges we can lift
        # this to a per-action getter.
        out = np.zeros(self.embodiment.action_dim(), dtype=np.float32)
        for i, afield in enumerate(self.embodiment.actions):
            if afield.name == "zero":
                out[i] = 0.0
            elif afield.name.startswith("d_"):
                # find matching state field
                source = afield.name[2:]
                for j, sfield in enumerate(self.embodiment.state):
                    if sfield.name == source:
                        out[i] = delta[j]
                        break
                else:
                    raise ValueError(
                        f"embodiment action {afield.name!r} has no matching "
                        f"state field {source!r}"
                    )
            else:
                raise ValueError(f"unknown action field {afield.name!r}")
        return out

    def _state_to_drone(self, traj: Trajectory, i: int) -> DroneState:
        ned = self.frame_graph.frame("ned")
        return DroneState(
            pos=Point(traj.positions_ned[i], frame=ned),
            vel=(traj.velocities_ned[i] if traj.velocities_ned is not None
                 else np.zeros(3)),
            quat_xyzw=traj.quaternions_xyzw[i],
            t=float(traj.times[i]),
        )

    @staticmethod
    def _encode_png(img: np.ndarray) -> bytes:
        from PIL import Image as _Image
        # PIL.fromarray treats 3-channel uint8 as RGB; since we pre-swapped
        # to BGR per the embodiment, the resulting PNG bytes hold BGR data
        # (PIL doesn't know — it just round-trips the buffer). That matches
        # DroneVLA2.0's cv2.imwrite output exactly.
        buf = io.BytesIO()
        _Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()


# State-field getters live in `falsify.policy.state_assembly.STATE_GETTERS`
# now — shared with `PiGatewayPolicy` and `VLAPolicy` so a single
# embodiment YAML drives state assembly in both train and eval pipelines.
