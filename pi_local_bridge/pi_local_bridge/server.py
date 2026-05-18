"""WSS handler speaking the pi-inference-client gateway wire protocol.

Implements the four message types the client sends — ``load``, ``infer``,
``reset``, ``telemetry`` — over a sync ``websockets`` server. Inference
dispatches into a locally-loaded ``pi_inference_client.local.LocalPolicyClient``.

Concurrency model
-----------------
JAX/CUDA inference is single-stream and the history-mode backend keeps
per-policy buffer state, so we serialize all ``infer`` calls behind an
``RLock``. Multiple concurrent connections are accepted but their
``infer``s are interleaved one-at-a-time. Each new connection bumps a
session counter and the policy's history buffer is reset, so two
overlapping clients don't corrupt each other's history.

Auth
----
A comma-separated allow-list of API keys is read from an env var named by
``auth.api_keys_env`` in the bridge YAML. Connections without an
``Authorization: Api-Key <key>`` header matching the allow-list are
rejected at handshake time with HTTP 401.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Heavy imports (jax via pi_inference_client.local) are deferred to
# `_build_state` so `--help` works in a falsify venv without the `local`
# extra installed.


logger = logging.getLogger("pi_local_bridge")


# ---------------------------------------------------------------------------
# Wire-protocol channel names — copied from pi_inference_client.client
# ---------------------------------------------------------------------------

_LOAD = "load"
_INFER = "infer"
_RESET = "reset"
_TELEMETRY = "telemetry"


# ---------------------------------------------------------------------------
# Config + state
# ---------------------------------------------------------------------------


@dataclass
class BridgeState:
    policy: Any                         # _PolicyHandle
    client: Any                         # LocalPolicyClient
    spec_json: str
    lock: threading.RLock
    ws_path: str
    api_keys: frozenset[str]


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

    # --- (1) image-key resolver ------------------------------------
    _orig_resolve = papi._resolve_image_key

    def _resolve(user_key, valid_fields):
        if user_key in valid_fields:
            return user_key
        return _orig_resolve(user_key, valid_fields)

    papi._resolve_image_key = _resolve

    # --- (2) image-shape parser ------------------------------------
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
            # Read the inner tuple block right after the `<field>:` line and
            # collect every leading numeric scalar. The shape is whatever
            # numbers appear inside one nested `!tuple` block.
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
            # Find trailing channel dim (3) and read H, W as the two preceding.
            if len(nums) >= 3 and nums[-1] == 3:
                image_shapes[field] = (nums[-3], nums[-2])
        return fields, image_shapes

    papi._extract_process_input_fields = _extract

    papi._BRIDGE_PATCHED = True
    logger.info(
        "patched pi_inference_client.local: _resolve_image_key (identity), "
        "_extract_process_input_fields (history-aware shape)"
    )


def _build_state(cfg: dict) -> BridgeState:
    # Imports that pull in jax happen here, not at module top-level, so
    # `python -m pi_local_bridge --help` works in a falsify venv too.
    _apply_local_api_patches()
    from pi_inference_client.local import LocalPolicyClient, load_policy
    from pi_local_bridge.spec import policy_spec_json

    policy_args = argparse.Namespace(**cfg["policy"])
    logger.info("loading checkpoint: %s", policy_args.ckpt_path)
    t0 = time.time()
    policy = load_policy(policy_args)
    client = LocalPolicyClient(policy)
    logger.info(
        "checkpoint loaded in %.1fs — action_horizon=%d action_dim=%d cameras=%s",
        time.time() - t0,
        policy.action_horizon,
        policy.action_dim,
        list(policy.image_key_map.values()),
    )
    spec_json = policy_spec_json(policy, spec_overrides=cfg.get("spec_overrides"))
    return BridgeState(
        policy=policy,
        client=client,
        spec_json=spec_json,
        lock=threading.RLock(),
        ws_path=cfg["listen"].get("ws_path", "/"),
        api_keys=_read_api_keys(cfg.get("auth") or {}),
    )


# ---------------------------------------------------------------------------
# Request handling
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
    """The client may send raw ndarray or JPEG bytes (compress_images=True)."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        from pi_inference_client.preprocessing import jpeg_decode
        return jpeg_decode(bytes(value))
    if isinstance(value, dict):
        # video_encode=True path — not supported in bridge yet.
        raise NotImplementedError(
            "Video-encoded image payload received; bridge supports raw ndarray "
            "or JPEG bytes only (set client `video_encode: false`)."
        )
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def _handle_load(state: BridgeState, _payload: object) -> bytes:
    return _ok({"result": {"spec": state.spec_json}})


def _reset_history(policy: Any) -> None:
    """Mirror of ``pi_inference_client.local.reset_history`` — inlined so the
    bridge dispatch path doesn't import jax through ``local.__init__``."""
    history = getattr(policy, "_history", None)
    if history is not None:
        history.reset()


def _handle_reset(state: BridgeState, _payload: object) -> bytes:
    with state.lock:
        _reset_history(state.policy)
        state.client.reset()
    return _ok({})


def _handle_telemetry(_state: BridgeState, _payload: object) -> bytes:
    # We accept telemetry but do nothing with it — the falsify side already
    # writes per-query debug bundles. Returning success keeps the client
    # from buffering forever and from logging "failed to send telemetry".
    return _ok({})


def _handle_infer(state: BridgeState, payload: object) -> bytes:
    # `pi_inference_client.local.InputData` is just a slots dataclass
    # with the duck-typed attrs `create_input` reads. We use SimpleNamespace
    # so this handler stays importable in environments without jax (helps
    # testing — production always has jax via the [local] extra).
    from types import SimpleNamespace

    if not isinstance(payload, dict):
        raise TypeError("infer payload must be a dict")
    inference_input = payload.get("inference_input") or {}
    images_raw = inference_input.get("image") or {}
    states_raw = inference_input.get("state") or {}
    inputs_block = inference_input.get("inputs") or {}

    # Build the local-side InputData. Local InputData expects the same
    # payload keys (server-side input names), but the local processor maps
    # them back via `policy.image_key_map`. We rebuild the *user-facing*
    # mapping by looking up each payload key against `image_key_map`.
    reverse_image = {v: k for k, v in state.policy.image_key_map.items()}
    reverse_state = {v: k for k, v in state.policy.state_key_map.items()}

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
        # Realtime fields are sent on every request by the client even when
        # RTC is off (defaults to zeros + a trivial prefix_info). Pass them
        # through verbatim — `_infer_with_realtime_fields` handles the
        # "no-op realtime" case gracefully when the exported method
        # supports the kwargs; otherwise call .infer().
        initial_noise = inference_input.get("initial_noise")
        prefix_info = inference_input.get("prefix_info")
        kwargs_spec = state.policy.exported_method.kwargs_spec
        use_rt = (
            initial_noise is not None
            and "initial_noise" in kwargs_spec
            and ("prefix_info" in kwargs_spec or "prefix" in kwargs_spec)
        )
        if use_rt:
            result = state.client._infer_with_realtime_fields(
                input_data,
                initial_noise=initial_noise,
                prefix_info=prefix_info,
                sample_args=sample_args,
            )
        else:
            result = state.client.infer(input_data)
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
# WS connection + auth
# ---------------------------------------------------------------------------


def _process_request_factory(state: BridgeState):
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def _process_request(ws, request):
        # Path check.
        if request.path.rstrip("/") != state.ws_path.rstrip("/"):
            return Response(404, "Not Found", Headers(), b"unknown ws path\n")
        # Auth check.
        if state.api_keys:
            authz = request.headers.get("Authorization", "")
            if not authz.startswith("Api-Key "):
                return Response(401, "Unauthorized", Headers(), b"missing Api-Key\n")
            key = authz[len("Api-Key "):].strip()
            if key not in state.api_keys:
                return Response(401, "Unauthorized", Headers(), b"unknown api key\n")
        return None  # accept upgrade
    return _process_request


def _connection_handler_factory(state: BridgeState):
    def _handle(ws) -> None:
        peer = ws.remote_address
        logger.info("client connected: %s", peer)
        with state.lock:
            _reset_history(state.policy)
            state.client.reset()
        try:
            for raw in ws:
                if isinstance(raw, str):
                    # Clients should always send binary frames; reject text.
                    ws.send(_err(ValueError("expected binary msgpack frame")))
                    continue
                response = _handle_message(state, raw)
                ws.send(response)
        except Exception:
            logger.exception("connection handler error")
        finally:
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
        "pi-local-bridge listening on ws://%s:%d%s (%d api key%s loaded)",
        host, port, state.ws_path,
        len(state.api_keys), "" if len(state.api_keys) == 1 else "s",
    )
    with serve(
        _connection_handler_factory(state),
        host=host,
        port=port,
        process_request=_process_request_factory(state),
        max_size=50 * 1024 * 1024,
    ) as server:
        server.serve_forever()
