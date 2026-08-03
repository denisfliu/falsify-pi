"""Standalone smoke test for the gate-drone pi0 checkpoint.

Reproduces gate_inference.GatePolicy's load path but pads the shipped 7-D norm
stats up to the model action_dim (32) so openpi's Normalize/Unnormalize (which
run at the padded model dimension) broadcast correctly. Padding with
mean=0 / std=1 / q=0..1 on dims 7..31 is numerically faithful: those state dims
are always the zero-padding LiberoInputs adds, and the padded action dims are
sliced off by LiberoOutputs before returning the 7-D control action.

Usage:
    python local/smoke_gate.py --ckpt local/checkpoints/gate_both_scratch \
                               --norm local/assets/gate_nav
"""
import argparse
import dataclasses
import numpy as np

import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.transforms import NormStats


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
        out[k] = NormStats(
            mean=ext(s.mean, 0.0), std=ext(s.std, 1.0),
            q01=ext(s.q01, 0.0), q99=ext(s.q99, 1.0),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--config", default="pi0_gate")
    a = ap.parse_args()

    cfg = _cfg.get_config(a.config)
    action_dim = cfg.model.action_dim
    print(f"config={a.config} action_dim={action_dim} "
          f"use_quantile_norm={cfg.data.create(__import__('pathlib').Path('.'), cfg.model).use_quantile_norm}")

    ns = _nz.load(a.norm)
    print("raw stat dims:", {k: int(np.asarray(v.mean).shape[-1]) for k, v in ns.items()})
    ns = _pad_stats(ns, action_dim)
    print("padded stat dims:", {k: int(np.asarray(v.mean).shape[-1]) for k, v in ns.items()})

    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)

    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    obs = {
        "observation/image": img,
        "observation/wrist_image": img.copy(),
        "observation/state": np.zeros(7, np.float32),
        "prompt": "go through the gate on the left and hover over the stuffed animal",
    }
    out = policy.infer(obs)
    act = np.asarray(out["actions"])
    print("SMOKE_OK actions", act.shape, "dtype", act.dtype,
          "range", float(act.min()), float(act.max()))
    print("control (x,y,z,yaw) of first 3 steps:\n", act[:3, :4])


if __name__ == "__main__":
    main()
