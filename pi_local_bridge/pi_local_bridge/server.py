"""WSS handler speaking the pi-inference-client gateway wire protocol.

Implements the four message types the client sends — ``load``, ``infer``,
``reset``, ``telemetry`` — over a sync ``websockets`` server. Inference
dispatches into a locally-loaded ``pi_inference_client.local.LocalPolicyClient``.

Multi-policy hosting
--------------------
The bridge holds a *registry* of policy entries (id → ckpt_path + processor +
wire-side config). Exactly one entry is resident in GPU memory at a time —
the "active" one. Each entry binds to its own ``ws_path``.

The active policy NEVER changes implicitly. Clients connecting to a
registered ws_path whose entry is not the active one are rejected with HTTP
**409 Conflict** and a body identifying the current active id. To swap, an
operator (or the falsify-side CLI) must GET ``/admin/switch_policy?policy_id=<id>``.
This guarantees that every run knows exactly which checkpoint produced its
actions. (GET — not POST — because the ``websockets`` sync server's HTTP
parser rejects non-GET methods before our ``process_request`` hook runs.
The mutation is idempotent and the endpoint is internal-only.)

Concurrency model
-----------------
JAX/CUDA inference is single-stream and the history-mode backend keeps
per-policy buffer state, so we serialize all ``infer`` calls behind an
``RLock``. A swap holds the same lock, closes all open WS connections (so
clients re-do ``load`` + reset history against the new policy), drops the
old handle, and loads the new one.

Auth
----
A comma-separated allow-list of API keys is read from an env var named by
``auth.api_keys_env`` in the bridge YAML. Both WS connections AND the admin
HTTP endpoints accept the same ``Authorization: Api-Key <key>`` header
against this allow-list.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Heavy imports (jax via pi_inference_client.local) are deferred to
# `_load_policy_entry` so `--help` works in a falsify venv without the
# `local` extra installed.


logger = logging.getLogger("pi_local_bridge")


# ---------------------------------------------------------------------------
# Wire-protocol channel names — copied from pi_inference_client.client
# ---------------------------------------------------------------------------

_LOAD = "load"
_INFER = "infer"
_RESET = "reset"
_TELEMETRY = "telemetry"


# ---------------------------------------------------------------------------
# Registry + state
# ---------------------------------------------------------------------------


@dataclass
class PolicyEntry:
    """One entry in the bridge's policy registry."""
    policy_id: str
    ws_path: str
    policy_cfg: dict           # the per-policy block (ckpt_path, processor_name, …)
    spec_overrides: dict | None = None


@dataclass
class ActivePolicy:
    """The currently-loaded policy handle. Populated on every swap."""
    policy_id: str
    policy: Any                # pi_inference_client.local._PolicyHandle
    client: Any                # LocalPolicyClient
    spec_json: str
    loaded_at: float           # epoch seconds


@dataclass
class BridgeState:
    registry: dict[str, PolicyEntry]    # id → entry
    ws_path_to_id: dict[str, str]       # ws_path → id (inverse index)
    active: ActivePolicy
    lock: threading.RLock               # serializes infer + swap
    api_keys: frozenset[str]
    open_connections: set = field(default_factory=set)
    connections_lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Config loading + registry build
# ---------------------------------------------------------------------------


def _load_bridge_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_api_keys(auth_cfg: dict) -> frozenset[str]:
    env_name = auth_cfg.get("api_keys_env")
    if env_name:
        raw = os.environ.get(env_name, "")
        keys = {k.strip() for k in raw.split(",") if k.strip()}
        if not keys:
            logger.warning(
                "auth.api_keys_env=%r is unset or empty — all connections will be rejected",
                env_name,
            )
        return frozenset(keys)
    # Inline list (discouraged — only for local smoke testing).
    inline = auth_cfg.get("api_keys") or []
    return frozenset(str(k) for k in inline)


def _build_registry(cfg: dict) -> tuple[dict[str, PolicyEntry], str]:
    """Resolve the bridge YAML into (registry, default_policy_id).

    Two YAML shapes are accepted:

    1. **Legacy single-policy** — top-level ``policy:`` block, ws_path taken
       from ``listen.ws_path``. Wrapped into a one-entry registry.
    2. **Multi-policy** — top-level ``policies:`` mapping id → block; each
       block carries its own ``ws_path``. ``default_policy:`` names the
       initial active id.
    """
    listen_ws_path = (cfg.get("listen") or {}).get("ws_path", "/")
    if "policies" in cfg:
        raw_registry = cfg["policies"] or {}
        if not isinstance(raw_registry, dict) or not raw_registry:
            raise ValueError("'policies:' must be a non-empty mapping")
        registry: dict[str, PolicyEntry] = {}
        seen_paths: dict[str, str] = {}
        for pid, block in raw_registry.items():
            if not isinstance(block, dict):
                raise ValueError(f"policies.{pid} must be a mapping")
            ws_path = block.get("ws_path")
            if not ws_path:
                raise ValueError(f"policies.{pid}: missing 'ws_path'")
            if ws_path in seen_paths:
                raise ValueError(
                    f"ws_path {ws_path!r} declared by both "
                    f"{seen_paths[ws_path]!r} and {pid!r}"
                )
            seen_paths[ws_path] = pid
            policy_cfg = {k: v for k, v in block.items()
                          if k not in ("ws_path", "spec_overrides")}
            registry[pid] = PolicyEntry(
                policy_id=pid,
                ws_path=ws_path,
                policy_cfg=policy_cfg,
                spec_overrides=block.get("spec_overrides"),
            )
        default_id = cfg.get("default_policy")
        if default_id is None:
            if len(registry) == 1:
                default_id = next(iter(registry))
            else:
                raise ValueError(
                    "multi-policy registry requires `default_policy:` "
                    "at top level"
                )
        if default_id not in registry:
            raise ValueError(
                f"default_policy={default_id!r} not in registry "
                f"({sorted(registry)})"
            )
        return registry, default_id

    # Legacy form.
    if "policy" not in cfg:
        raise ValueError(
            "bridge YAML must declare either 'policies:' (multi) or "
            "'policy:' (single)"
        )
    pid = (cfg.get("policy") or {}).get("processor_name") or "default"
    entry = PolicyEntry(
        policy_id=pid,
        ws_path=listen_ws_path,
        policy_cfg=cfg["policy"],
        spec_overrides=cfg.get("spec_overrides"),
    )
    return {pid: entry}, pid


# ---------------------------------------------------------------------------
# Upstream patches (unchanged from v1)
# ---------------------------------------------------------------------------


def _apply_local_api_patches() -> None:
    """Monkey-patches that make `pi_inference_client.local` usable with the
    dronevla v7 gate-scenes checkpoints.

    Two upstream issues exist for this checkpoint family:

    1. `_resolve_image_key` assumes camera-suffixed payload names
       (`observation/<camera>/rgb/image`). The v7 base camera is just
       `observation/rgb/image` (no camera segment), which the heuristic
       can't construct. We let any user_key resolve to itself if it
       already names a valid processor input.

    2. `_extract_process_input_fields` parses image shapes with a regex
       fixed at three numeric components, treating the leading
       history-sequence dim of v7's `(seq, H, W, C)` shape as the height.
       Result: `image_h=6, image_w=448`. We replace the parser with one
       that walks the inner tuple, finds the trailing 3-channel
       component, and returns (H, W) — falling back to the original
       behaviour if the shape isn't recognizably image-like.

    Both patches are additive (no behaviour change for already-supported
    checkpoint layouts) and idempotent.
    """
    import re
    import pi_inference_client.local.policy_api as papi
    if getattr(papi, "_BRIDGE_PATCHED", False):
        return

    _orig_resolve = papi._resolve_image_key

    def _resolve(user_key, valid_fields):
        if user_key in valid_fields:
            return user_key
        return _orig_resolve(user_key, valid_fields)

    papi._resolve_image_key = _resolve

    def _extract(process_spec_path):
        text = process_spec_path.read_text()
        input_fields = papi._extract_spec_section_fields(text, "inputs")
        fields = set(input_fields)
        image_shapes: dict[str, tuple[int, int]] = {}
        for field in fields:
            if not (
                field.startswith("observation/images/")
                or (field.startswith("observation/") and field.endswith("/rgb/image"))
            ):
                continue
            field_re = re.compile(
                rf"^\s{{2}}{re.escape(field)}:\s*!tuple\s*\n"
                r"\s*-\s*!tuple\s*\n"
                r"((?:\s*-\s*\d+\s*\n)+)",
                flags=re.MULTILINE,
            )
            block = field_re.search(text)
            if not block:
                continue
            nums = [int(n) for n in re.findall(r"-\s*(\d+)", block.group(1))]
            if len(nums) >= 3 and nums[-1] == 3:
                image_shapes[field] = (nums[-3], nums[-2])
        return fields, image_shapes

    papi._extract_process_input_fields = _extract

    papi._BRIDGE_PATCHED = True
    logger.info(
        "patched pi_inference_client.local: _resolve_image_key (identity), "
        "_extract_process_input_fields (history-aware shape)"
    )


# ---------------------------------------------------------------------------
# Policy load / swap
# ---------------------------------------------------------------------------


def _load_policy_entry(entry: PolicyEntry) -> ActivePolicy:
    """Materialize a PolicyEntry into a loaded ActivePolicy."""
    _apply_local_api_patches()
    from pi_inference_client.local import LocalPolicyClient, load_policy
    from pi_local_bridge.spec import policy_spec_json

    policy_args = argparse.Namespace(**entry.policy_cfg)
    logger.info("[%s] loading checkpoint: %s",
                entry.policy_id, getattr(policy_args, "ckpt_path", "?"))
    t0 = time.time()
    policy = load_policy(policy_args)
    client = LocalPolicyClient(policy)
    logger.info(
        "[%s] checkpoint loaded in %.1fs — action_horizon=%d action_dim=%d cameras=%s",
        entry.policy_id, time.time() - t0,
        policy.action_horizon, policy.action_dim,
        list(policy.image_key_map.values()),
    )
    spec_json = policy_spec_json(policy, spec_overrides=entry.spec_overrides)
    return ActivePolicy(
        policy_id=entry.policy_id,
        policy=policy,
        client=client,
        spec_json=spec_json,
        loaded_at=time.time(),
    )


def _close_all_connections(state: BridgeState) -> int:
    """Close every open WS connection. Returns the count closed."""
    with state.connections_lock:
        conns = list(state.open_connections)
        state.open_connections.clear()
    n = 0
    for ws in conns:
        try:
            ws.close(code=1012, reason="bridge swapping active policy")
            n += 1
        except Exception:
            logger.debug("error closing connection during swap", exc_info=True)
    return n


def _swap_policy(state: BridgeState, target_id: str) -> ActivePolicy:
    """Replace the active policy under the bridge lock.

    Returns the new ActivePolicy. Caller MUST already hold or be able to
    acquire state.lock — we re-enter it here to serialize with infer.
    """
    if target_id not in state.registry:
        raise KeyError(
            f"policy_id={target_id!r} not in registry "
            f"({sorted(state.registry)})"
        )
    with state.lock:
        if state.active.policy_id == target_id:
            logger.info("[swap] %s already active — no-op", target_id)
            return state.active
        old_id = state.active.policy_id
        logger.info("[swap] %s → %s …", old_id, target_id)
        kicked = _close_all_connections(state)
        if kicked:
            logger.info("[swap] kicked %d open connection(s)", kicked)
        # Drop the old handle and clear JAX's compile cache so the new
        # policy's tracing doesn't reuse stale shapes. GPU memory may not
        # fully release (XLA retains some pools); operators are expected
        # to restart the process if accumulated swaps drive OOM.
        try:
            old = state.active
            state.active = None  # type: ignore[assignment]
            del old
            gc.collect()
            try:
                import jax
                jax.clear_caches()
            except Exception:
                logger.debug("jax.clear_caches() failed", exc_info=True)
        except Exception:
            logger.exception("error tearing down old policy")
        new_active = _load_policy_entry(state.registry[target_id])
        state.active = new_active
        logger.info("[swap] %s → %s complete", old_id, target_id)
        return new_active


def _build_state(cfg: dict) -> BridgeState:
    registry, default_id = _build_registry(cfg)
    ws_path_to_id = {e.ws_path: e.policy_id for e in registry.values()}
    api_keys = _read_api_keys(cfg.get("auth") or {})
    logger.info(
        "registry: %d policy/policies, default=%s, ws_paths=%s",
        len(registry), default_id, sorted(ws_path_to_id),
    )
    active = _load_policy_entry(registry[default_id])
    return BridgeState(
        registry=registry,
        ws_path_to_id=ws_path_to_id,
        active=active,
        lock=threading.RLock(),
        api_keys=api_keys,
    )


# ---------------------------------------------------------------------------
# WS dispatch (unchanged contract — operates on state.active)
# ---------------------------------------------------------------------------


def _ok(payload: object) -> bytes:
    from pi_inference_client import msgpack_numpy
    return msgpack_numpy.packb(({"success": True}, payload))


def _err(exc: BaseException) -> bytes:
    from pi_inference_client import msgpack_numpy
    return msgpack_numpy.packb((
        {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
        None,
    ))


def _decode_image(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        from pi_inference_client.preprocessing import jpeg_decode
        return jpeg_decode(bytes(value))
    if isinstance(value, dict):
        raise NotImplementedError(
            "Video-encoded image payload received; bridge supports raw ndarray "
            "or JPEG bytes only (set client `video_encode: false`)."
        )
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def _handle_load(state: BridgeState, _payload: object) -> bytes:
    return _ok({"result": {"spec": state.active.spec_json}})


def _reset_history(policy: Any) -> None:
    history = getattr(policy, "_history", None)
    if history is not None:
        history.reset()


def _handle_reset(state: BridgeState, _payload: object) -> bytes:
    with state.lock:
        _reset_history(state.active.policy)
        state.active.client.reset()
    return _ok({})


def _handle_telemetry(_state: BridgeState, _payload: object) -> bytes:
    return _ok({})


def _handle_infer(state: BridgeState, payload: object) -> bytes:
    from types import SimpleNamespace

    if not isinstance(payload, dict):
        raise TypeError("infer payload must be a dict")
    inference_input = payload.get("inference_input") or {}
    images_raw = inference_input.get("image") or {}
    states_raw = inference_input.get("state") or {}
    inputs_block = inference_input.get("inputs") or {}

    active = state.active
    reverse_image = {v: k for k, v in active.policy.image_key_map.items()}
    reverse_state = {v: k for k, v in active.policy.state_key_map.items()}

    images: dict[str, np.ndarray] = {}
    for payload_key, raw in images_raw.items():
        if payload_key.endswith("_mask"):
            continue
        user_key = reverse_image.get(payload_key, payload_key)
        images[user_key] = _decode_image(raw)

    states: dict[str, np.ndarray] = {}
    for payload_key, raw in states_raw.items():
        user_key = reverse_state.get(payload_key, payload_key)
        states[user_key] = np.asarray(raw, dtype=np.float32)

    robot_task = inputs_block.get("robot_task_string")
    raw_text = inputs_block.get("raw_text")
    sample_args = payload.get("sample_args") or inference_input.get("sample_args")

    input_data = SimpleNamespace(
        images=images,
        states=states,
        robot_task_string=robot_task,
        raw_text=raw_text,
        conditioning=None,
    )

    t0 = time.perf_counter()
    with state.lock:
        initial_noise = inference_input.get("initial_noise")
        prefix_info = inference_input.get("prefix_info")
        kwargs_spec = active.policy.exported_method.kwargs_spec
        use_rt = (
            initial_noise is not None
            and "initial_noise" in kwargs_spec
            and ("prefix_info" in kwargs_spec or "prefix" in kwargs_spec)
        )
        if use_rt:
            result = active.client._infer_with_realtime_fields(
                input_data,
                initial_noise=initial_noise,
                prefix_info=prefix_info,
                sample_args=sample_args,
            )
        else:
            result = active.client.infer(input_data)
    infer_ms = (time.perf_counter() - t0) * 1000

    outputs: dict[str, Any] = dict(result.actions_by_key)
    if not outputs:
        outputs["action/action"] = np.asarray(result.actions)
    raw_outputs: dict[str, Any] = {}
    if result.raw_actions is not None:
        raw_outputs["actions"] = np.asarray(result.raw_actions)

    response_payload = {
        "result": {"outputs": outputs, "raw_outputs": raw_outputs},
        "inference_time_ms": infer_ms,
        "server_total_time_ms": infer_ms,
    }
    return _ok(response_payload)


_DISPATCH = {
    _LOAD: _handle_load,
    _INFER: _handle_infer,
    _RESET: _handle_reset,
    _TELEMETRY: _handle_telemetry,
}


def _handle_message(state: BridgeState, raw: bytes) -> bytes:
    from pi_inference_client import msgpack_numpy
    try:
        unpacked = msgpack_numpy.unpackb(raw, raw=False)
    except Exception as exc:
        return _err(exc)
    if not isinstance(unpacked, (list, tuple)) or len(unpacked) != 2:
        return _err(ValueError("malformed request frame"))
    api, payload = unpacked
    handler = _DISPATCH.get(api)
    if handler is None:
        return _err(ValueError(f"unknown api: {api!r}"))
    try:
        return handler(state, payload)
    except Exception as exc:
        logger.exception("handler error in %s", api)
        return _err(exc)


# ---------------------------------------------------------------------------
# HTTP admin routes (intercepted before the WS upgrade)
# ---------------------------------------------------------------------------


def _admin_policies_body(state: BridgeState) -> bytes:
    body = {
        "active_policy_id": state.active.policy_id,
        "active_loaded_at": state.active.loaded_at,
        "policies": [
            {
                "policy_id": pid,
                "ws_path": entry.ws_path,
                "ckpt_path": entry.policy_cfg.get("ckpt_path"),
                "processor_name": entry.policy_cfg.get("processor_name"),
                "is_active": pid == state.active.policy_id,
            }
            for pid, entry in state.registry.items()
        ],
    }
    return json.dumps(body, indent=2).encode("utf-8") + b"\n"


def _handle_admin_switch(state: BridgeState, target_id: str | None) -> tuple[int, bytes]:
    if not target_id:
        return 400, b"missing 'policy_id' query parameter\n"
    if target_id not in state.registry:
        return 404, (
            f"unknown policy_id={target_id!r}; "
            f"known={sorted(state.registry)}\n"
        ).encode("utf-8")
    try:
        new_active = _swap_policy(state, target_id)
    except Exception as exc:
        logger.exception("swap failed")
        return 500, f"swap failed: {exc}\n".encode("utf-8")
    body_out = json.dumps({
        "active_policy_id": new_active.policy_id,
        "active_loaded_at": new_active.loaded_at,
    }).encode("utf-8") + b"\n"
    return 200, body_out


# ---------------------------------------------------------------------------
# WS connection + auth
# ---------------------------------------------------------------------------


def _check_auth(headers) -> tuple[bool, str]:
    """Returns (ok, reason). On ok=True reason is the matched key.
    Caller checks against `state.api_keys` itself — here we just parse."""
    authz = headers.get("Authorization", "")
    if not authz.startswith("Api-Key "):
        return False, "missing Api-Key"
    key = authz[len("Api-Key "):].strip()
    if not key:
        return False, "empty Api-Key"
    return True, key


def _process_request_factory(state: BridgeState):
    from urllib.parse import parse_qs, urlsplit
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def _process_request(ws, request):
        parsed = urlsplit(request.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query) if parsed.query else {}
        # ---- Admin HTTP routes ----------------------------------------
        if path.startswith("/admin/"):
            ok, key_or_reason = _check_auth(request.headers)
            if not ok or key_or_reason not in state.api_keys:
                msg = f"admin auth failed: {key_or_reason if not ok else 'unknown api key'}\n"
                return Response(401, "Unauthorized", Headers({"Content-Type": "text/plain"}),
                                msg.encode("utf-8"))
            if path == "/admin/policies":
                body = _admin_policies_body(state)
                return Response(200, "OK", Headers({"Content-Type": "application/json"}), body)
            if path == "/admin/switch_policy":
                target_id = (query.get("policy_id") or [None])[0]
                code, body_out = _handle_admin_switch(state, target_id)
                reason = "OK" if code == 200 else "Error"
                return Response(code, reason,
                                Headers({"Content-Type": "application/json" if code == 200 else "text/plain"}),
                                body_out)
            return Response(404, "Not Found", Headers(), b"unknown admin route\n")

        # ---- WS upgrade path ------------------------------------------
        if path not in {p.rstrip("/") for p in state.ws_path_to_id}:
            return Response(404, "Not Found", Headers(), b"unknown ws path\n")
        # Auth.
        if state.api_keys:
            ok, key_or_reason = _check_auth(request.headers)
            if not ok or key_or_reason not in state.api_keys:
                msg = f"ws auth failed: {key_or_reason if not ok else 'unknown api key'}\n"
                return Response(401, "Unauthorized", Headers({"Content-Type": "text/plain"}),
                                msg.encode("utf-8"))
        # Active-policy gate.
        requested_id = None
        for entry_path, pid in state.ws_path_to_id.items():
            if entry_path.rstrip("/") == path:
                requested_id = pid
                break
        if requested_id != state.active.policy_id:
            body = (
                f"policy_id={requested_id!r} is registered but not active; "
                f"current active={state.active.policy_id!r}. "
                f"GET /admin/switch_policy?policy_id={requested_id} "
                f"to switch.\n"
            ).encode("utf-8")
            return Response(409, "Conflict",
                            Headers({"Content-Type": "text/plain"}), body)
        return None  # accept upgrade
    return _process_request


def _connection_handler_factory(state: BridgeState):
    def _handle(ws) -> None:
        peer = ws.remote_address
        logger.info("client connected: %s (active=%s)", peer, state.active.policy_id)
        with state.connections_lock:
            state.open_connections.add(ws)
        with state.lock:
            _reset_history(state.active.policy)
            state.active.client.reset()
        try:
            for raw in ws:
                if isinstance(raw, str):
                    ws.send(_err(ValueError("expected binary msgpack frame")))
                    continue
                response = _handle_message(state, raw)
                ws.send(response)
        except Exception:
            logger.exception("connection handler error")
        finally:
            with state.connections_lock:
                state.open_connections.discard(ws)
            logger.info("client disconnected: %s", peer)
    return _handle


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_server(config_path: Path) -> None:
    from websockets.sync.server import serve

    cfg = _load_bridge_config(config_path)
    state = _build_state(cfg)
    listen = cfg["listen"]
    host = listen.get("host", "0.0.0.0")
    port = int(listen.get("port", 8765))

    logger.info(
        "pi-local-bridge listening on %s:%d (%d api key%s, %d polic%s registered, active=%s)",
        host, port,
        len(state.api_keys), "" if len(state.api_keys) == 1 else "s",
        len(state.registry), "y" if len(state.registry) == 1 else "ies",
        state.active.policy_id,
    )
    with serve(
        _connection_handler_factory(state),
        host=host,
        port=port,
        process_request=_process_request_factory(state),
        max_size=50 * 1024 * 1024,
    ) as server:
        server.serve_forever()
