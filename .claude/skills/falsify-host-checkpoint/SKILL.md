---
name: falsify-host-checkpoint
description: Host a Pi dronevla v7 checkpoint on this machine via `pi_local_bridge`. Downloads the checkpoint from `gs://dronevla-raw-data/...`, installs `pi-inference-client[local]` into a dedicated JAX venv, applies the upstream patches for v7's processor quirks, and launches a WSS server that speaks Pi's gateway protocol so any falsify-side `PiGatewayPolicy` (or raw `pi_inference_client.PolicyClient`) can connect by URL alone.
---

# falsify-host-checkpoint

Self-host the pi-inference-client gateway protocol against a checkpoint
we run locally (instead of routing through Pi's commercial gateway).
Companion to [`falsify-infer-from-checkpoint`](../falsify-infer-from-checkpoint/SKILL.md),
which connects from the falsify side.

## When to use

- We have a v7 dronevla finetune checkpoint in `gs://dronevla-raw-data/...`
  and want to run inference on our own GPU box.
- Pi hasn't deployed (or won't deploy) this specific checkpoint behind
  their commercial gateway, but the falsify rollout needs it now.
- Smoke-testing a new checkpoint variant before requesting a Pi-hosted
  route.

Not for: Pi-hosted gateways (`wss://api.pi-fleet.com/...` is the API
key holder's responsibility — just point `gateway_url` at it and skip
this skill).

## Inputs

- A checkpoint GCS URI (e.g., `gs://dronevla-raw-data/model_checkpoints/2026_may17/mar16_history_ft_dronevla_v7_gate_scenes_may16_311am/h6jtbq0w/20000`).
- Local disk space matching the checkpoint size (~7.5 GiB for v7 history).
- The PI partner GCP service-account key. Default path:
  `~/code/dataset_validation/pi-data-sharing/pi-external-partners-*.json`.
- A throwaway API key string (anything starting with `pi-`) for the
  bridge's `Authorization: Api-Key` header.

## Outputs

- A running `pi-local-bridge` server on `ws://HOST:PORT/v1/models/<name>`.
- A bridge YAML under `pi_local_bridge/configs/<name>.yaml` capturing
  the exact load_policy args used (reproducible).

## Procedure

### 1. Authenticate with the partner SA key (one-time per shell)

```bash
SA_KEY="${PI_INFERENCE_SA_KEY:-$HOME/code/dataset_validation/pi-data-sharing/pi-external-partners-*.json}"
gcloud auth activate-service-account --key-file=$(ls $SA_KEY | head -1)
gcloud config set account dronevla-external-sa@pi-external-partners.iam.gserviceaccount.com
```

`tools/pi_inference_env.sh` automates this and also exports
`$PI_GCP_ACCESS_TOKEN` and `$PI_PYTHON_INDEX_URL` for the pip install
step.

### 2. Create the bridge venv (one-time per machine)

**Important: separate venv from falsify's.** JAX 0.5+ / flax / orbax
clash with falsify's torch + nerfstudio + cu121 stack. The bridge has
its own `pyproject.toml` precisely to allow this isolation.

```bash
python3.11 -m venv ~/venvs/pi-local-bridge
source ~/venvs/pi-local-bridge/bin/activate
pip install --upgrade pip
```

### 3. Install `pi-inference-client[local]` + the bridge

```bash
TOKEN=$(gcloud auth print-access-token)
pip install \
  --extra-index-url "https://oauth2accesstoken:${TOKEN}@us-east5-python.pkg.dev/pi-external-partners/pi-python/simple/" \
  -e /home/dfliu/code/falsify/pi_local_bridge \
  "pi-inference-client[local]>=0.4.9"
```

### 4. Install jax CUDA wheels (one-time, ~700 MB)

The default jax install is CPU-only. Without CUDA each inference takes
~50 s on a 24-core CPU instead of sub-second on a GPU.

```bash
pip install --upgrade "jax[cuda12]==0.10.0"
```

Pin the version to match the meta `jax` already in the venv — mixing
versions across `jax`, `jaxlib`, `jax-cuda12-pjrt`, `jax-cuda12-plugin`
causes XLA initialization failures. Driver requirement is CUDA 12.x;
any NVIDIA driver new enough for CUDA 12 works (`nvidia-smi` shows
"CUDA Version: 12.x" or higher).

### 5. Download the checkpoint

```bash
mkdir -p /home/dfliu/checkpoints/dronevla_v7/<run_id>
gsutil -m cp -r \
  gs://dronevla-raw-data/model_checkpoints/2026_may17/.../<run_id>/<step> \
  /home/dfliu/checkpoints/dronevla_v7/<run_id>/
```

The dest dir parent must pre-exist (`mkdir -p` above) or gsutil's
`cp -r` rejects it with the "destination URL must name a directory"
error.

Verify the processor lands where `load_policy` expects
(`<ckpt_path>/processors/<processor_name>/`):

```bash
ls /home/dfliu/checkpoints/dronevla_v7/<run_id>/<step>/model_bfloat16/processors/
# → dronevla_v7_gate_scenes/   ← processor_name in bridge YAML
```

### 6. Write a bridge config YAML

Start from `pi_local_bridge/configs/local_v7_history.yaml` or
`v7_history.example.yaml`. The critical fields:

```yaml
listen:
  host: 0.0.0.0                # 127.0.0.1 for local-only smoke
  port: 8765                   # 8766 for the nonhistory variant if both run
  ws_path: /v1/models/v7-history

auth:
  api_keys_env: PI_BRIDGE_API_KEYS   # comma-separated allow-list

policy:
  ckpt_path: /home/dfliu/checkpoints/dronevla_v7/<run_id>/<step>/model_bfloat16
  processor_name: dronevla_v7_gate_scenes
  backend: pi07
  inference_function: sample_actions_fixed_noise
  initial_noise_mode: numpy_fixed_random
  initial_noise_seed: 0
  sampling_kwargs: { num_denoising_steps: 10 }
  action_horizon: 50
  execute_chunk_size: 25
  resize_with_pad: true
  sequence_length: 6           # 6 for history, 1 for nonhistory
  sequence_length_stride: 1
  processor_overrides: { text_sequence_length: 300, max_num_actions: 0 }
  # Use server-side keys directly. The bridge's _resolve_image_key patch
  # makes these resolve identity-style — needed because the upstream
  # heuristic can't handle the base-cam name `observation/rgb/image`.
  image_keys:
    - observation/rgb/image
    - observation/wrist_image/rgb/image
  state_keys: [observation/state]
  state_dims: { observation/state: 7 }

spec_overrides:
  image_preprocess:
    target_resolution: [448, 448]
    resize_mode: pad
    interpolation: bilinear
```

### 7. Launch the bridge

```bash
export PI_BRIDGE_API_KEYS="pi-some-throwaway-key,pi-another-key"
source ~/venvs/pi-local-bridge/bin/activate
python -m pi_local_bridge \
  --config pi_local_bridge/configs/local_v7_history.yaml \
  --log-level INFO
```

Successful startup sequence:

```
loading checkpoint: …/model_bfloat16
patched pi_inference_client.local: _resolve_image_key (identity), _extract_process_input_fields (history-aware shape)
Found a processor with a graph that expects DALI compressed images, …
[process=0] /jax/checkpoint/read/gbytes_per_sec: 2.1 GiB/s …
checkpoint loaded in 5.3s — action_horizon=50 action_dim=32 cameras=['observation/rgb/image', 'observation/wrist_image/rgb/image']
pi-local-bridge listening on ws://0.0.0.0:8765/v1/models/v7-history (2 api keys loaded)
```

`action_dim=32` here is the raw model output width. The actual
`actions_by_key['action/action']` returned to clients is `(chunk, 7)`
after the processor's unprocess step. This is expected — v7's pi07
backbone is a 32-D action-space model retrained for the 7-D drone task.

### 8. Connect with the smoke client to confirm

In the same bridge venv (so you have `pi-inference-client` and
`websockets>=12`):

```bash
PI_API_KEY=pi-some-throwaway-key \
  python pi_local_bridge/scripts/smoke_client.py \
    --url ws://127.0.0.1:8765/v1/models/v7-history \
    --n-iters 2
```

Expected:

```
connected + load handshake in 0.01s
server cameras: ['observation/rgb/image', 'observation/wrist_image/rgb/image']
action_horizon: 50, action_dim: 32, image_resolution: (448, 448)
infer #1   actions.shape: (25, 7)   raw_actions.shape: (50, 32)
infer #2   actions.shape: (25, 7)   raw_actions.shape: (50, 32)
```

Round-trip should be sub-second once CUDA jaxlib is installed. ~50 s
means JAX fell back to CPU — re-check step 4.

## Two known upstream issues the bridge patches automatically

These bite anyone trying to `load_policy()` a v7 gate-scenes checkpoint
naïvely — `server._apply_local_api_patches()` handles them at startup.
See `pi_local_bridge/README.md` for the rationale.

1. `_resolve_image_key` can't construct payload key `observation/rgb/image`
   because its heuristic assumes a non-empty camera segment between
   `observation/` and `/rgb/image`. Patch: identity passthrough when the
   user_key already matches a valid processor field.
2. `_extract_process_input_fields` reads the first three numbers of an
   image shape as `(H, W, C)`, but history-mode shapes are
   `(seq, H, W, C)` — so it returns `(image_h=6, image_w=448)`. Patch:
   walk the inner tuple, find the trailing channel `3`, return the two
   preceding values.

## Troubleshooting

- **"All connections rejected"** — `$PI_BRIDGE_API_KEYS` is unset or
  empty. Check `echo $PI_BRIDGE_API_KEYS` in the same shell launching
  the bridge.
- **"jax.errors.XlaRuntimeError: INTERNAL"** — jaxlib version
  mismatched with jax-cuda12-plugin. Reinstall with
  `pip install --upgrade "jax[cuda12]==<pinned version>"`.
- **`KeyError: 'observation/state'`** — the client used the singular
  `state=` kwarg; that always emits under `observation/joint_position`,
  which v7 doesn't accept. Use plural `states={'observation/state': vec}`
  or update the falsify-side `PiGatewayConfig.state_key` (default
  `'observation/state'` is correct).
- **`No module named 'jax'`** when running `--help`** — wrong venv.
  Activate `~/venvs/pi-local-bridge/` before running the bridge.
- **`gsutil … destination URL must name a directory`** — the dest dir
  parent must pre-exist for `cp -r`. `mkdir -p` it first.

## After hosting

Falsify-side configs in `configs/policies/pi_gateway/*.yaml` reference
`gateway_url: ws://moraband.stanford.edu:PORT/...`. For a local bridge
on the same machine as falsify, edit those to point at
`ws://127.0.0.1:PORT/...` (or pass `--policy-config` to
`falsify.cli.run_vla_episode` pointing at a copy with the URL changed).
