"""Synthesize a PolicySpec JSON from a locally-loaded Pi policy handle.

The Pi gateway responds to ``load`` with a JSON-encoded ``PolicySpec`` that the
client uses to discover camera keys, action keys, action horizon, image
preprocessing contract, etc. ``pi_inference_client.local`` doesn't expose this
spec directly (it builds a runtime ``ServerConfig`` instead), so we re-derive
it here from a ``_PolicyHandle`` plus a YAML-supplied override block.
"""

from __future__ import annotations

import json
from typing import Any


def build_policy_spec(
    policy: Any,                       # pi_inference_client.local.policy_api._PolicyHandle
    *,
    spec_overrides: dict | None = None,
) -> dict:
    """Build a PolicySpec dict matching what the hosted gateway publishes."""
    overrides = dict(spec_overrides or {})

    # ---- input_spec --------------------------------------------------
    # Format: input_spec[key] = [shape, dtype]. The client's
    # `_first_shape` walks until it finds a tuple/list of ints, so any
    # nested representation works as long as the leaf shape ends in 3 for
    # images.
    input_spec: dict[str, Any] = {}

    img_h, img_w = int(policy.image_h), int(policy.image_w)
    for falsify_key, payload_key in policy.image_key_map.items():
        input_spec[payload_key] = [[img_h, img_w, 3], "uint8"]
        # The client tags an `<image_key>_mask` boolean alongside each image
        # when the server declares one. The local backend already wires the
        # mask internally, so we declare it for symmetry.
        input_spec[f"{payload_key}_mask"] = [[], "bool_"]

    for falsify_key, payload_key in policy.state_key_map.items():
        dim = int(policy.state_dims.get(falsify_key, 0))
        if dim <= 0:
            # Best-effort fallback: the local backend will still accept any
            # 1-D state and the processor handles slicing.
            dim = 7
        input_spec[payload_key] = [[dim], "float32"]

    # ---- output_spec / action_keys / horizons ------------------------
    action_keys = list(policy.action_keys or [])
    output_spec: dict[str, Any] = {}
    if action_keys:
        for key in action_keys:
            output_spec[key] = [
                [int(policy.action_horizon), int(policy.action_dim)],
                "float32",
            ]
    else:
        output_spec["action/action"] = [
            [int(policy.action_horizon), int(policy.action_dim)],
            "float32",
        ]

    # ---- image_preprocess --------------------------------------------
    image_preprocess = {
        "target_resolution": [img_h, img_w],
        "resize_mode": "pad" if policy.resize_with_pad else "stretch",
        "interpolation": "bilinear",
    }
    if "image_preprocess" in overrides:
        image_preprocess.update(overrides.pop("image_preprocess"))

    # ---- final spec --------------------------------------------------
    spec: dict[str, Any] = {
        "input_spec": input_spec,
        "output_spec": output_spec,
        "action_horizon": int(policy.action_horizon),
        "action_dim": int(policy.action_dim),
        "image_preprocess": image_preprocess,
    }
    if action_keys:
        spec["action_keys"] = action_keys

    # Apply any remaining top-level overrides last (lets the YAML force any
    # field we got wrong without us having to ship a patch).
    spec.update(overrides)
    return spec


def policy_spec_json(policy: Any, *, spec_overrides: dict | None = None) -> str:
    return json.dumps(build_policy_spec(policy, spec_overrides=spec_overrides))
