"""Serve a gate-drone pi0 checkpoint over openpi's websocket protocol.

Mirrors openpi's scripts/serve_policy.py but (a) uses the reconstructed `pi0_gate`
config and (b) pads the shipped 7-D norm stats up to the model action_dim (32) so
Normalize/Unnormalize broadcast correctly (see local/smoke_gate.py for the rationale).
falsify's VLAPolicy connects as the client (openpi_client protocol).

Run in the openpi venv:
    env -u VIRTUAL_ENV ~/code/openpi/.venv/bin/python local/serve_gate.py \
        --ckpt local/checkpoints/gate_both_scratch --norm local/assets/gate_nav --port 8000
"""
import argparse
import numpy as np

import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.transforms import NormStats
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def _pad_stats(ns: dict[str, NormStats], dim: int) -> dict[str, NormStats]:
    out = {}
    for k, s in ns.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s
            continue
        pad = dim - n
        def ext(arr, fill):
            return None if arr is None else np.concatenate(
                [np.asarray(arr, np.float32), np.full(pad, fill, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0.0), std=ext(s.std, 1.0),
                           q01=ext(s.q01, 0.0), q99=ext(s.q99, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-prompt", default=None)
    a = ap.parse_args()

    cfg = _cfg.get_config(a.config)
    ns = _pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(
        cfg, a.ckpt, norm_stats=ns, default_prompt=a.default_prompt)
    print(f"[serve_gate] policy loaded (config={a.config} action_dim={cfg.model.action_dim}); "
          f"serving on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(policy, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
