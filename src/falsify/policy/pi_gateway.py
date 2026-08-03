"""Pi-inference-client gateway policy.

Bridges falsify's frame-tagged ``Observation`` / ``Trajectory`` contract to
Pi's hosted inference gateway (``pi_inference_client.PolicyClient``). This
is the successor to ``VLAPolicy`` for the dronevla v7 gate-scenes
finetunes — single base + single wrist camera (no 3pov), 30 Hz, 7-D state,
7-D action, server-driven image resolution.

Frame contract
--------------
Same as ``VLAPolicy``: the finetuned policies inherit the dronevla v7 frame
convention (MOCAP-z-up, perm5 NED↔MOCAP, negated yaw on state). Frame
conversions happen exactly once per chunk inside this module — no frame
leaks. Position deltas in MOCAP are integrated cumulatively, prepended
with the current pose, then converted back to NED.

What's configurable (YAML, not code)
------------------------------------
- ``gateway_url``           — per-checkpoint Pi gateway endpoint.
- ``api_key``               — supports ``${env:PI_API_KEY}`` indirection.
- ``execute_chunk_size``    — number of actions executed per request.
- ``hz``                    — control rate; v7 gate-scenes models are 30 Hz.
- ``camera_map``            — falsify modality (``forward``/``downward``)
                              → server camera key (e.g.
                              ``observation/image`` /
                              ``observation/wrist_image``). Server-declared
                              ``camera_names`` validates this at connect.
- ``state_dim``/``action_dim`` — input + output widths (default 7/7).
- ``action_pos_slice`` / ``action_yaw_index`` — which action columns are
                              position deltas vs. yaw delta. The v7 model
                              has dims 4–6 constant zero, so defaults are
                              ``[0:3]`` and ``3``.
- ``server_frame``          — frame the policy was trained in (default
                              ``"mocap"``).
- ``traceability``          — free-form metadata block (W&B run id, step,
                              GCS checkpoint path, processor name); never
                              load-bearing, just written next to each
                              debug bundle for reproducibility.
- ``record_dir``            — optional debug bundle directory.

The Pi client is imported lazily so falsify keeps loading on machines that
have not yet installed ``pi-inference-client``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from falsify.geometry import FrameGraph, Point, Trajectory, assert_frame

from .base import Policy
from .camera_postprocess import CameraPostprocess
from .observation import Observation
from .state_assembly import build_state_vector


_ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)\}")


def _expand_env(value: str | None) -> str | None:
    """Expand ``${env:VAR}`` placeholders in a string. Identity on None."""
    if value is None:
        return None
    def _sub(m: re.Match[str]) -> str:
        var = m.group(1)
        v = os.environ.get(var)
        if v is None:
            raise RuntimeError(f"PiGatewayPolicy: env var {var!r} is unset")
        return v
    return _ENV_RE.sub(_sub, value)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PiGatewayConfig:
    # --- Connection ---
    gateway_url: str = ""
    api_key: str = ""                       # supports ${env:PI_API_KEY}
    execute_chunk_size: int = 25
    connect_open_timeout: float = 360.0
    compress_images: bool = True
    jpeg_quality: int = 85
    sample_args: Optional[dict] = None

    # --- Task ---
    prompt: str = ""

    # --- Control + integration ---
    hz: int = 30
    state_dim: int = 7
    action_dim: int = 7
    # Action layout — which columns to integrate.
    action_pos_slice: tuple[int, int] = (0, 3)   # [start, stop)
    action_yaw_index: Optional[int] = 3           # None ⇒ ignore yaw

    # --- Modality wiring ---
    # falsify camera name → server camera key (server.camera_names).
    # For the v7 gate-scenes finetune: forward → observation/rgb/image,
    # downward → observation/wrist_image/rgb/image.
    camera_map: dict[str, str] = field(default_factory=lambda: {
        "forward":  "observation/rgb/image",
        "downward": "observation/wrist_image/rgb/image",
    })
    # Pre-server image preprocessing. Both knobs default to "send what
    # the renderer produced" (legacy behaviour) but the v7 gate-scenes
    # finetunes need to be told that the training data was authored at
    # 256² and channel-flipped to BGR by `falsify.training.exporter`. If
    # we don't replicate both at eval, the model receives
    # color-swapped + different-aspect images.
    #
    # `image_size`     — if set, resize each cam's native render to this
    #                    square edge (PIL bilinear) before sending. The
    #                    server still applies its own preprocess on top
    #                    (v7 = 448² square resize). Set to `null` to skip.
    # `channel_order`  — "RGB" (default) sends the renderer's RGB pixels
    #                    verbatim. "BGR" flips channels to match training
    #                    data that was exported with `channel_order: BGR`.
    image_size: Optional[int] = None
    channel_order: str = "RGB"

    # Per-camera RGBA overlay paths (e.g. drone gripper baked onto the
    # downward camera). Keys are falsify camera names ("forward",
    # "downward"); paths are relative to the working directory or
    # absolute. Cameras not in this dict get no overlay. Composited as
    # the last step of preprocess so train/eval see identical bytes.
    gripper_overlay_paths: dict[str, str] = field(default_factory=dict)

    # Server-side state input key. The Pi client's default `state=` kwarg
    # always emits `observation/joint_position`; the v7 server expects
    # `observation/state` instead, so we route the state vector via the
    # plural `states=` mapping with this explicit key.
    state_key: str = "observation/state"

    # --- Frame ---
    server_frame: str = "mocap"

    # Path to the embodiment YAML the trained checkpoint was authored
    # against. When set, the policy walks `embodiment.state` to build
    # the state vector — same code path as `TrainingDataExporter`. When
    # `None`, falls back to the hardcoded v7 layout
    # `[px_mocap, py_mocap, pz_mocap, -yaw_ned, 0, 0, 0]`. Setting this
    # is the recommended path because it makes "train and eval send the
    # same state shape" a property of the YAMLs, not of policy code.
    embodiment_path: Optional[str] = None

    # --- Bridge admin handshake (pi_local_bridge multi-policy hosting) ---
    # When `bridge_admin_url` is set, the policy adapter GETs
    # /admin/switch_policy?policy_id=<bridge_policy_id> on the bridge
    # before opening the WS connection. This is the *explicit* assertion
    # that the bridge is serving the checkpoint this YAML describes — if
    # the bridge can't switch, the run aborts. Both fields must be set
    # together; leave both unset for Pi-hosted gateway URLs where there
    # is no admin endpoint to talk to.
    bridge_admin_url: Optional[str] = None
    bridge_policy_id: Optional[str] = None
    bridge_switch_timeout: float = 360.0    # JAX cold load is ~30-60s

    # --- Optional RTC (deferred; off by default) ---
    use_rtc: bool = False

    # --- Traceability (metadata only, not load-bearing) ---
    traceability: dict[str, Any] = field(default_factory=dict)

    # --- Debug ---
    record_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # YAML loader — single source of truth for `pi_gateway`-typed
    # policy YAMLs under `configs/policies/pi_gateway/`. The CLIs
    # (`run_vla_episode`, `run_eval_campaign`,
    # `collect_recovery_trajectories`, `smoke_test`) all funnel through
    # this so a new YAML field requires one edit here, not four.
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict,
        *,
        prompt_override: Optional[str] = None,
        execute_chunk_size_override: Optional[int] = None,
        use_rtc_override: Optional[bool] = None,
        record_dir: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> "PiGatewayConfig":
        """Build a `PiGatewayConfig` from a parsed YAML dict.

        Validates ``type: pi_gateway`` if the key is present. Caller
        overrides win over the YAML — used by the eval campaign /
        recovery drivers, which thread CLI flags through the same loader.

        ``source`` is a free-form tag (typically the YAML path) used
        only in error messages.
        """
        if cfg.get("type") not in (None, "pi_gateway"):
            tag = f" from {source}" if source else ""
            raise ValueError(
                f"PiGatewayConfig.from_dict{tag}: expected type=pi_gateway, "
                f"got {cfg.get('type')!r}"
            )
        record_dir_resolved = record_dir
        if record_dir_resolved is None and cfg.get("record_dir") is not None:
            record_dir_resolved = Path(cfg["record_dir"])
        return cls(
            gateway_url=cfg["gateway_url"],
            api_key=cfg.get("api_key", ""),
            execute_chunk_size=int(
                execute_chunk_size_override
                if execute_chunk_size_override is not None
                else cfg.get("execute_chunk_size", 25)
            ),
            prompt=prompt_override if prompt_override is not None else cfg.get("prompt", ""),
            hz=int(cfg.get("hz", 30)),
            state_dim=int(cfg.get("state_dim", 7)),
            action_dim=int(cfg.get("action_dim", 7)),
            action_pos_slice=tuple(cfg.get("action_pos_slice", (0, 3))),
            action_yaw_index=cfg.get("action_yaw_index", 3),
            camera_map=dict(cfg.get("camera_map") or {}),
            state_key=cfg.get("state_key", "observation/state"),
            server_frame=cfg.get("server_frame", "mocap"),
            bridge_admin_url=cfg.get("bridge_admin_url"),
            bridge_policy_id=cfg.get("bridge_policy_id"),
            use_rtc=bool(use_rtc_override if use_rtc_override is not None
                         else cfg.get("use_rtc", False)),
            image_size=cfg.get("image_size"),
            channel_order=str(cfg.get("channel_order", "RGB")),
            gripper_overlay_paths=dict(cfg.get("gripper_overlay_paths") or {}),
            sample_args=cfg.get("sample_args"),
            traceability=dict(cfg.get("traceability") or {}),
            record_dir=record_dir_resolved,
            embodiment_path=cfg.get("embodiment_path"),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        prompt_override: Optional[str] = None,
        execute_chunk_size_override: Optional[int] = None,
        use_rtc_override: Optional[bool] = None,
        record_dir: Optional[Path] = None,
    ) -> "PiGatewayConfig":
        """Load + parse a `pi_gateway`-typed YAML file. Thin wrapper around `from_dict`."""
        import yaml as _yaml
        path = Path(path)
        cfg = _yaml.safe_load(path.read_text())
        return cls.from_dict(
            cfg,
            prompt_override=prompt_override,
            execute_chunk_size_override=execute_chunk_size_override,
            use_rtc_override=use_rtc_override,
            record_dir=record_dir,
            source=str(path),
        )


# ---------------------------------------------------------------------------
# Helpers (mirroring VLAPolicy's conventions)
# ---------------------------------------------------------------------------


from falsify.geometry import (
    quat_to_yaw_xyzw as _quat_to_yaw_xyzw,
    yaw_to_quat_xyzw as _yaw_to_quat_xyzw,
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PiGatewayPolicy(Policy):
    """Pi inference-client gateway-backed VLA policy.

    Falsify-side wrapper around ``pi_inference_client.PolicyClient`` that
    enforces the NED↔MOCAP boundary and adapts our typed ``Observation`` to
    Pi's ``InputData`` schema. Lazy on the Pi import so falsify keeps
    loading without the wheel installed.
    """

    def __init__(self, cfg: PiGatewayConfig, frame_graph: FrameGraph) -> None:
        self.cfg = cfg
        self.frame_graph = frame_graph
        self._client = None         # pi_inference_client.PolicyClient
        self._server_config = None  # cached pi_inference_client.ServerConfig
        self._rtc_runner = None     # pi_inference_client.AsyncRTCPolicyRunner (only when use_rtc)
        self._connected = False
        self._query_count = 0
        self._step_count = 0
        # Populated by `_bridge_admin_handshake` on first connect when
        # `bridge_admin_url` is set. CLIs persist this into
        # policy_manifest.json so a reviewer can prove which bridge
        # checkpoint produced the run.
        self.bridge_manifest: Optional[dict[str, Any]] = None

        # Per-camera postprocess pipeline (resize → channel swap →
        # optional overlay). Same `CameraPostprocess` the training
        # exporter uses; sharing the class makes train/eval byte-parity
        # a property of the code, not of YAML-keeping discipline.
        unknown_overlays = set(cfg.gripper_overlay_paths) - set(cfg.camera_map)
        if unknown_overlays:
            raise ValueError(
                f"PiGatewayConfig.gripper_overlay_paths references cameras "
                f"not in camera_map: {sorted(unknown_overlays)}"
            )
        self._postprocess: dict[str, CameraPostprocess] = {
            cam_name: CameraPostprocess.from_paths(
                image_size=cfg.image_size,
                channel_order=cfg.channel_order,
                overlay_path=cfg.gripper_overlay_paths.get(cam_name),
            )
            for cam_name in cfg.camera_map
        }

        # Optional embodiment-driven state assembly. Loaded lazily here
        # so falsify.training is only imported when this knob is set —
        # keeps mock-policy runs from pulling pyarrow.
        self._embodiment = None
        if cfg.embodiment_path is not None:
            from falsify.training import load_embodiment
            self._embodiment = load_embodiment(cfg.embodiment_path)
            if self._embodiment.state_dim() != cfg.state_dim:
                raise ValueError(
                    f"PiGatewayPolicy: embodiment {cfg.embodiment_path!r} declares "
                    f"state_dim={self._embodiment.state_dim()} but config.state_dim="
                    f"{cfg.state_dim} — these must match"
                )

    @property
    def required_modalities(self) -> frozenset[str]:
        return frozenset(f"images.{name}" for name in self.cfg.camera_map)

    def reset(self) -> None:
        self._query_count = 0
        self._step_count = 0
        # Clear server-side state at the episode boundary. The bridge only
        # resets a history checkpoint's frame buffer (`policy._history`) on a
        # wire `reset` message or a fresh WS connection — never on its own. So
        # when a single PiGatewayPolicy/connection is reused across episodes,
        # the previous episode's tail frames would otherwise bleed into the
        # next episode's history window. No-op until the lazy connection is
        # established (a fresh connection resets history bridge-side anyway).
        if not self._connected:
            return
        if self._rtc_runner is not None:
            # Also clears the RTC denoising-prefix accumulation locally, then
            # forwards the wire reset to the bridge (reset_server=True default).
            self._rtc_runner.reset()
        elif self._client is not None:
            self._client.reset()

    # ---- handshake ----------------------------------------------------

    def _bridge_admin_handshake(self) -> None:
        """Ensure the bridge is serving `bridge_policy_id` before we connect.

        GETs /admin/switch_policy?policy_id=<id>. The bridge no-ops if
        the policy is already active. Non-2xx ⇒ raise so the run aborts
        before any inference happens.
        """
        import json as _json
        import urllib.error as _ue
        import urllib.parse as _up
        import urllib.request as _ur
        if not self.cfg.bridge_admin_url or not self.cfg.bridge_policy_id:
            return
        if self.cfg.bridge_admin_url and not self.cfg.bridge_policy_id:
            raise ValueError(
                "PiGatewayPolicy: bridge_admin_url is set but bridge_policy_id "
                "is empty — set both or neither"
            )
        api_key = _expand_env(self.cfg.api_key) or ""
        qs = _up.urlencode({"policy_id": self.cfg.bridge_policy_id})
        url = self.cfg.bridge_admin_url.rstrip("/") + "/admin/switch_policy?" + qs
        # GET (not POST) — the `websockets` sync server's HTTP parser
        # rejects non-GET requests before our process_request hook runs.
        # Idempotent + internal-only, so GET is fine here.
        req = _ur.Request(url, method="GET")
        req.add_header("Authorization", f"Api-Key {api_key}")
        t0 = time.time()
        try:
            with _ur.urlopen(req, timeout=self.cfg.bridge_switch_timeout) as resp:
                body = resp.read()
                status = resp.status
        except _ue.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"PiGatewayPolicy: bridge swap to {self.cfg.bridge_policy_id!r} "
                f"failed (HTTP {exc.code}): {body_txt.strip()}"
            ) from exc
        except _ue.URLError as exc:
            raise RuntimeError(
                f"PiGatewayPolicy: cannot reach bridge admin "
                f"{self.cfg.bridge_admin_url!r}: {exc}"
            ) from exc
        elapsed = time.time() - t0
        try:
            parsed = _json.loads(body)
        except _json.JSONDecodeError:
            raise RuntimeError(
                f"PiGatewayPolicy: bridge swap returned non-JSON body "
                f"(HTTP {status}): {body!r}"
            )
        active = parsed.get("active_policy_id")
        if active != self.cfg.bridge_policy_id:
            raise RuntimeError(
                f"PiGatewayPolicy: bridge swap returned active={active!r}, "
                f"expected {self.cfg.bridge_policy_id!r}"
            )
        self.bridge_manifest = {
            "bridge_admin_url": self.cfg.bridge_admin_url,
            "bridge_policy_id": self.cfg.bridge_policy_id,
            "switch_took_s": elapsed,
            "active_policy_id": active,
            "active_loaded_at": parsed.get("active_loaded_at"),
        }
        print(
            f"[pi_gateway] bridge active={active} "
            f"(switch took {elapsed:.1f}s)"
        )

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._bridge_admin_handshake()
        from pi_inference_client import ClientConfig, PolicyClient  # noqa: WPS433
        cc = ClientConfig(
            gateway_url=_expand_env(self.cfg.gateway_url),
            api_key=_expand_env(self.cfg.api_key),
            execute_chunk_size=self.cfg.execute_chunk_size,
            robot_task_string=self.cfg.prompt,
            sample_args=self.cfg.sample_args,
            connect_open_timeout=self.cfg.connect_open_timeout,
            compress_images=self.cfg.compress_images,
            jpeg_quality=self.cfg.jpeg_quality,
        )
        self._client = PolicyClient(cc)
        self._client.connect()
        self._server_config = self._client.server_config
        self._validate_server_config()
        if self.cfg.use_rtc:
            from pi_inference_client import AsyncRTCConfig, AsyncRTCPolicyRunner  # noqa: WPS433
            rtc_cfg = AsyncRTCConfig(control_hz=float(self.cfg.hz))
            self._rtc_runner = AsyncRTCPolicyRunner(self._client, rtc_cfg)
        self._connected = True

    def _validate_server_config(self) -> None:
        """Cross-check the server's declared camera_names against our map."""
        if self._server_config is None:
            return
        declared = set(self._server_config.camera_names or [])
        wanted = set(self.cfg.camera_map.values())
        unknown = wanted - declared if declared else set()
        if unknown:
            raise RuntimeError(
                f"PiGatewayPolicy: camera_map references keys the server "
                f"does not declare: {sorted(unknown)} "
                f"(server camera_names = {sorted(declared)})"
            )

    def close(self) -> None:
        if self._rtc_runner is not None:
            try:
                self._rtc_runner.close()
            except Exception:
                pass
            self._rtc_runner = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False

    # ---- main loop ----------------------------------------------------

    def observe(self, obs: Observation) -> Trajectory:
        from pi_inference_client import InputData  # noqa: WPS433

        # 1) NED → server frame (MOCAP).
        pos_state = obs.state.pos
        assert_frame(pos_state, "ned")
        pos_mocap = self.frame_graph.convert(pos_state, to=self.cfg.server_frame).xyz
        yaw_ned = _quat_to_yaw_xyzw(obs.state.quat_xyzw)
        yaw_to_vla = -yaw_ned

        # 2) Build the state vector. When an embodiment YAML is wired
        #    via `cfg.embodiment_path`, walk its declared state schema
        #    (same code path as `TrainingDataExporter`). Otherwise fall
        #    back to the hardcoded v7 gate-scenes layout
        #    `[px_mocap, py_mocap, pz_mocap, -yaw_ned, 0, 0, 0]`.
        if self._embodiment is not None:
            pos_ned = pos_state.xyz
            state_vec = build_state_vector(
                self._embodiment,
                pos_mocap=pos_mocap, yaw_mocap=yaw_to_vla,
                pos_ned=pos_ned, yaw_ned=yaw_ned,
            )
        else:
            state_vec = np.zeros(self.cfg.state_dim, dtype=np.float32)
            state_vec[0:3] = pos_mocap
            if self.cfg.state_dim >= 4:
                state_vec[3] = yaw_to_vla

        # 3) Images — runs the shared CameraPostprocess (resize → channel
        #    swap → optional gripper overlay) so what we send is
        #    byte-identical to what `falsify.training.exporter` wrote at
        #    training time. The server's own image_preprocess (v7 = 448²
        #    square resize) still runs on top.
        images: dict[str, np.ndarray] = {}
        native_images: dict[str, np.ndarray] = {}
        sent_images: dict[str, np.ndarray] = {}
        for cam_name, server_key in self.cfg.camera_map.items():
            rgb_native = obs.require(f"images.{cam_name}")
            native_images[cam_name] = rgb_native
            sent = self._postprocess[cam_name].apply(rgb_native)
            images[server_key] = sent
            sent_images[cam_name] = sent

        prompt = obs.prompt or self.cfg.prompt
        if not prompt:
            raise RuntimeError(
                "PiGatewayPolicy: empty prompt. The orchestrator must inject "
                "a prompt via obs.prompt (resolved from a trial card's "
                "`prompt_name` against configs/prompts/atomic_dataset_prompts"
                ".yaml), or the caller must set PiGatewayConfig.prompt. "
                "Refusing to send an empty robot_task_string — the policy "
                "was trained on specific task strings and a blank prompt "
                "would silently degrade inference."
            )
        # Use `states=` (plural) so the state lands under the configured
        # server-side key. The default `state=` kwarg always routes to
        # `observation/joint_position`, which the v7 server doesn't accept.
        input_data = InputData(
            images=images,
            states={self.cfg.state_key: state_vec},
            robot_task_string=prompt,
            sample_args=dict(self.cfg.sample_args) if self.cfg.sample_args else None,
        )

        # 4) Query.
        self._ensure_connected()
        t0 = time.time()
        if self.cfg.use_rtc:
            # RTC mode: runner.act(obs) returns ONE action per call. The
            # runner amortizes inference across `steps_between_inference`
            # ticks and uses the prior chunk's tail as the denoising
            # prefix. Caller is expected to run with chunk_steps=1.
            rt_result = self._rtc_runner.act(input_data)
            actions = np.atleast_2d(np.asarray(rt_result.action, dtype=np.float64))
            raw_actions_array = None
        else:
            result = self._client.infer(input_data)
            actions = np.asarray(result.actions, dtype=np.float64)
            raw_actions_array = (
                np.asarray(result.raw_actions, dtype=np.float64)
                if getattr(result, "raw_actions", None) is not None else None
            )
        infer_s = time.time() - t0
        if actions.ndim != 2 or actions.shape[1] < (self.cfg.action_pos_slice[1]):
            raise ValueError(
                f"PiGatewayPolicy: server returned actions shape {actions.shape}; "
                f"expected (N, ≥{self.cfg.action_pos_slice[1]})"
            )

        # 5) Integrate position deltas in MOCAP. RTC and chunk modes share
        # the same integration — RTC just hands us a length-1 chunk.
        max_n = 1 if self.cfg.use_rtc else self.cfg.execute_chunk_size
        n = min(actions.shape[0], max_n)
        ps, pe = self.cfg.action_pos_slice
        deltas = actions[:n, ps:pe]
        positions_mocap = np.cumsum(deltas, axis=0) + pos_mocap[None, :]
        positions_mocap = np.concatenate(
            [pos_mocap[None, :], positions_mocap], axis=0,
        )

        # 6) Yaw integration (carry NED yaw; subtract MOCAP yaw delta — same
        #    sign-flip convention as VLAPolicy).
        yaws_ned = np.zeros(n + 1)
        yaws_ned[0] = yaw_ned
        if self.cfg.action_yaw_index is not None and actions.shape[1] > self.cfg.action_yaw_index:
            yaw_deltas_mocap = actions[:n, self.cfg.action_yaw_index]
            for i, dy in enumerate(yaw_deltas_mocap):
                yaws_ned[i + 1] = yaws_ned[i] - float(dy)
        else:
            yaws_ned[:] = yaw_ned
        quats_xyzw = np.stack([_yaw_to_quat_xyzw(y) for y in yaws_ned], axis=0)

        times = np.arange(positions_mocap.shape[0]) / self.cfg.hz + obs.state.t

        # 7) Convert back to NED.
        server_frame = self.frame_graph.frame(self.cfg.server_frame)
        traj_mocap_pos = Trajectory(
            times=times, positions=positions_mocap, frame=server_frame,
        )
        traj_ned_pos = self.frame_graph.convert(traj_mocap_pos, to="ned")
        ned_frame = self.frame_graph.frame("ned")
        traj_ned = Trajectory(
            times=times,
            positions=traj_ned_pos.positions,
            frame=ned_frame,
            quaternions=quats_xyzw,
        )
        assert_frame(traj_ned, "ned")

        # 8) Debug recording.
        if self.cfg.record_dir is not None:
            self._record_query(
                pos_state.xyz, yaw_ned, pos_mocap, state_vec, prompt,
                native_images, sent_images, actions, traj_ned, infer_s,
                raw_actions_array,
            )
        self._query_count += 1
        self._step_count += n
        return traj_ned

    # ---- debug recording ----------------------------------------------

    def _record_query(
        self,
        pos_ned_xyz: np.ndarray,
        yaw_ned: float,
        pos_mocap: np.ndarray,
        state_vec_to_vla: np.ndarray,
        prompt: str,
        native_images: dict[str, np.ndarray],
        sent_images: dict[str, np.ndarray],
        actions: np.ndarray,
        traj_ned: Trajectory,
        infer_seconds: float,
        raw_actions: Optional[np.ndarray],
    ) -> None:
        from PIL import Image as _Image
        root = Path(self.cfg.record_dir)
        qdir = root / f"query_{self._query_count:04d}_step_{self._step_count:05d}"
        qdir.mkdir(parents=True, exist_ok=True)

        for cam_name, rgb in native_images.items():
            _Image.fromarray(rgb).save(qdir / f"rgb_{cam_name}.png")
        # Also dump the post-preprocess images that we actually handed to
        # the server — so the channel-order / resize transform is visible
        # in the debug bundle. PIL doesn't know about BGR; if
        # channel_order=BGR these PNGs hold BGR bytes labeled RGB, exactly
        # matching the training-data on-disk format.
        for cam_name, rgb in sent_images.items():
            _Image.fromarray(rgb).save(qdir / f"sent_{cam_name}.png")

        np.save(qdir / "actions.npy", actions)
        if raw_actions is not None:
            np.save(qdir / "raw_actions.npy", raw_actions)
        np.save(qdir / "waypoints_ned.npy", traj_ned.positions)

        meta = {
            "query_index": self._query_count,
            "step_count_so_far": self._step_count,
            "infer_seconds": infer_seconds,
            "prompt": prompt,
            "state_ned_pos": pos_ned_xyz.tolist(),
            "state_ned_yaw_rad": yaw_ned,
            "pos_mocap": pos_mocap.tolist(),
            "state_vec_to_vla": state_vec_to_vla.tolist(),
            "actions_shape": list(actions.shape),
            "use_rtc": self.cfg.use_rtc,
            "traceability": self.cfg.traceability,
        }
        (qdir / "data.json").write_text(json.dumps(meta, indent=2))
