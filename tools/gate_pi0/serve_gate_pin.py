"""Serve the gate-drone pi0 **pin** variant over openpi's websocket protocol.

Same as serve_gate.py (padded norm stats + WebsocketPolicyServer) but wraps the
policy so each inference injects the pinned source noise instead of sampling:

    ms    = policy._input_transform(obs)["state"]        # 32-D model state
    c     = MLP([ms, onehot(prompt)])                    # (K,)  instruction prior
    g     = N(0, 1) of shape (H, AD)
    noise = g - (g·U)Uᵀ + (c·Uᵀ)                         # pin g's U-subspace to c
    actions = policy.infer(obs, noise=noise)

Requires the openpi source-noise patch (Policy.infer + pi0.sample_actions accept
`noise=`). Run in the openpi venv:

    env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false ~/code/openpi/.venv/bin/python \
        local/serve_gate_pin.py --ckpt local/checkpoints/gate_both_pin \
        --norm local/assets/gate_nav --pin-u local/assets/pin_U_gate_k5.npy \
        --prior local/assets/prior_gate_mlp.pt --port 8000
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

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


class PinPolicy:
    """Duck-typed openpi policy: computes the instruction prior + pinned noise
    per query and forwards to the underlying (padded) Pi0 policy."""

    def __init__(self, policy, pin_u_path, prior_path):
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)          # (H*AD, K)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        self.tasks = d["tasks"]
        self.H, self.AD, self.K = d["H"], d["AD"], d["K"]
        layers, din = [], d["in_dim"]
        for h in d["hidden"]:
            layers += [nn.Linear(din, h), nn.SiLU()]; din = h
        layers += [nn.Linear(din, d["K"])]
        self.prior = nn.Sequential(*layers)
        self.prior.load_state_dict(d["state_dict"])
        self.prior.eval()
        self._rng = np.random.default_rng()

    def _onehot(self, prompt):
        p = str(prompt).lower()
        v = np.zeros(len(self.tasks), np.float32)
        for i, t in enumerate(self.tasks):
            key = "left" if "left" in str(t).lower() else ("right" if "right" in str(t).lower() else None)
            if key and key in p:
                v[i] = 1.0
        if v.sum() == 0:
            v[0] = 1.0
        return v

    def infer(self, obs: dict) -> dict:
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        oh = self._onehot(obs.get("prompt", ""))
        with torch.no_grad():
            x = torch.tensor(np.concatenate([ms, oh])[None].astype(np.float32))
            c = self.prior(x)[0].numpy()                          # (K,)
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32)
        gf = g.reshape(-1)
        noise = (gf - (gf @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD)
        return self.policy.infer(obs, noise=noise.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-prompt", default=None)
    a = ap.parse_args()

    cfg = _cfg.get_config(a.config)
    ns = _pad_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(
        cfg, a.ckpt, norm_stats=ns, default_prompt=a.default_prompt)
    pin = PinPolicy(policy, a.pin_u, a.prior)
    print(f"[serve_gate_pin] pin policy loaded (config={a.config} K={pin.K} "
          f"tasks={len(pin.tasks)}); serving on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
