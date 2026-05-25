"""Minimal end-to-end smoke for a running pi-local-bridge instance.

Connects via `pi_inference_client.PolicyClient`, calls `load` to fetch the
server spec, then runs a single fake-frames `infer`. Prints timings and the
shape of the returned action chunk. Designed to be run from the bridge venv
(which has `pi-inference-client` + `websockets>=12`) — pointing at a
locally-listening bridge.

Run:

    PI_API_KEY=pi-local-smoke-key \\
    /home/dfliu/venvs/pi-local-bridge/bin/python \\
        pi_local_bridge/scripts/smoke_client.py \\
        --url ws://127.0.0.1:8765/v1/models/v7-history
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from pi_inference_client import ClientConfig, InputData, PolicyClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/v1/models/v7-history")
    ap.add_argument("--api-key-env", default="PI_API_KEY")
    ap.add_argument("--prompt", required=True,
                    help="Task prompt to send. Use a string that exists in the "
                         "checkpoint's training data — see "
                         "configs/prompts/atomic_dataset_prompts.yaml in falsify.")
    ap.add_argument("--n-iters", type=int, default=2,
                    help="run >1 to amortize the JAX compile cost across calls")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"ERROR: ${args.api_key_env} not set", file=sys.stderr)
        return 2

    cfg = ClientConfig(
        gateway_url=args.url,
        api_key=api_key,
        execute_chunk_size=25,
        robot_task_string=args.prompt,
    )
    print(f"connecting to {args.url} …", flush=True)
    t0 = time.time()
    with PolicyClient(cfg) as client:
        client.connect()
        print(f"connected + load handshake in {time.time()-t0:.2f}s")
        srv = client.server_config
        print("server cameras:    ", srv.camera_names)
        print("action_horizon:    ", srv.action_horizon)
        print("action_dim:        ", srv.action_dim)
        print("image_resolution:  ", srv.image_resolution)
        print("image_preprocess:  ",
              {"resize_mode": srv.image_preprocess.resize_mode,
               "interpolation": srv.image_preprocess.interpolation,
               "target_resolution": srv.image_preprocess.target_resolution}
              if srv.image_preprocess else None)

        # Build a fake observation matching the v7 input spec — base + wrist,
        # 7-D state. Pass native (non-448) sizes; the client resizes to the
        # server's preprocess contract.
        h, w = 480, 640
        rgb = (np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))

        for i in range(args.n_iters):
            # Server expects state under `observation/state` (7-D), not the
            # default `observation/joint_position` that the singular `state=`
            # kwarg implies. Use plural `states=` with the explicit key.
            obs = InputData(
                images={
                    "observation/rgb/image":              rgb,
                    "observation/wrist_image/rgb/image":  rgb,
                },
                states={"observation/state": np.zeros(7, dtype=np.float32)},
                robot_task_string=args.prompt,
            )
            print(f"\n--- infer #{i+1} ---")
            t0 = time.time()
            result = client.infer(obs)
            dt = time.time() - t0
            print(f"client.infer round-trip: {dt:.2f}s")
            print(f"  actions.shape:     {result.actions.shape}")
            print(f"  actions_by_key:    {list(result.actions_by_key.keys())}")
            if result.raw_actions is not None:
                print(f"  raw_actions.shape: {result.raw_actions.shape}")
            if result.metrics is not None:
                m = result.metrics
                print(f"  metrics.server_processing_time_ms: {m.server_processing_time_ms}")
                print(f"  metrics.network_time_ms:           {m.network_time_ms}")

        client.reset()
        print("\nreset OK")
    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
