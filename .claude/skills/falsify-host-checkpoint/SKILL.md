---
name: falsify-host-checkpoint
description: Run a self-hosted `pi_local_bridge` WSS server that speaks Pi's gateway protocol against one or more locally-loaded dronevla v7 checkpoints. Covers single-policy bring-up, the multi-policy registry, adding a policy, swapping the active policy via the admin endpoint, and how falsify-side `PiGatewayPolicy` handshakes with the bridge to guarantee the right checkpoint is loaded before a run.
---

# falsify-host-checkpoint

Self-host the pi-inference-client gateway protocol against checkpoints we
run on our own hardware (instead of routing through Pi's commercial
gateway). The bridge can hold a **registry of N policies** behind one
port and rotate which one is GPU-resident on demand, so a single bridge
process serves every dronevla v7 finetune we have downloaded.

Companion to [`falsify-infer-from-checkpoint`](../falsify-infer-from-checkpoint/SKILL.md),
which connects from the falsify side.

## When to use

- We have one or more v7 dronevla finetune checkpoints in
  `gs://dronevla-raw-data/...` and want to run inference on our own GPU
  box (moraband, the falsify dev box, etc.).
- We want to A/B between several checkpoints without spinning up a
  bridge process per variant.
- We need a reviewer to be able to prove **which** checkpoint produced
  any given run — the bridge + falsify-side handshake make this an
  unambiguous property of the policy YAML.

Not for: Pi-hosted gateways (`wss://api.pi-fleet.com/...` is the
API-key holder's responsibility — just point `gateway_url` at it and
skip this skill).

## Architecture in one picture

```
moraband (GPU host)                          falsify CLI host
─────────────────────────────────            ─────────────────────────
pi-local-bridge (one process)                python -m falsify.cli.run_vla_episode
  ├─ listen 0.0.0.0:8765                          │
  ├─ registry: {                                  │
  │     v7-history       → ws_path A,             │
  │     v7-nonhistory    → ws_path B,             │
  │     v7-nh-all        → ws_path C, …}          │
  ├─ active=v7-history                            │
  └─ GET /admin/switch_policy?policy_id=…  ◄──────┤ 1. handshake (HTTP)
                                                  │   POSTs go here first
       ws://…/v1/models/v7-nonhistory  ◄──────────┤ 2. WS connect (post-swap)
                                                  │   policy.observe() → infer
```

Two contracts make this safe:
1. **No implicit swap** — connecting to a registered ws_path whose policy
   isn't active returns HTTP 409. You cannot accidentally infer against
   the wrong checkpoint.
2. **Explicit handshake** — `PiGatewayPolicy._ensure_connected` calls
   the admin endpoint before opening the WS. The YAML is the single
   source of truth for "which checkpoint this run uses." Non-2xx ⇒ the
   run aborts before any inference.

## Inputs

- A GCS URI for each checkpoint you want to host (e.g.,
  `gs://dronevla-raw-data/model_checkpoints/2026_may19/.../<wandb_run>/<step>`).
- Local disk space matching the checkpoint sizes (~7.5 GiB per v7
  finetune).
- The PI partner GCP service-account key (for `gsutil` + the private pip
  index). Default path:
  `~/code/dataset_validation/pi-data-sharing/pi-external-partners-*.json`.
- One or more throwaway API key strings (anything starting with `pi-`)
  for the bridge's `Authorization: Api-Key` header.

## Outputs

- A running `pi-local-bridge` process on `ws://<host>:8765/...` exposing
  N registered ws_paths plus `GET /admin/policies` and
  `GET /admin/switch_policy?policy_id=X`.
- A bridge registry YAML under `pi_local_bridge/configs/<name>.yaml`
  capturing the exact `load_policy` args for each entry (reproducible).
- Per-falsify-run `policy_manifest.json` files capturing the policy YAML
  sha256 + bridge handshake response.

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

### 5. Download checkpoints

The dest dir parent must pre-exist (`mkdir -p` first) or `gsutil cp -r`
errors with "destination URL must name a directory."

```bash
# Example: pi07 history (h6jtbq0w / step 20000)
mkdir -p /scratch/checkpoints/dronevla_v7/h6jtbq0w
gsutil -m cp -r \
  gs://dronevla-raw-data/model_checkpoints/2026_may17/mar16_history_ft_dronevla_v7_gate_scenes_may16_311am/h6jtbq0w/20000 \
  /scratch/checkpoints/dronevla_v7/h6jtbq0w/

# Example: pi07 nonhistory (ccvhs1do / step 20000)
mkdir -p /scratch/checkpoints/dronevla_v7/ccvhs1do
gsutil -m cp -r \
  gs://dronevla-raw-data/model_checkpoints/2026_may17/mar16_nonhistory_ft_dronevla_v7_gate_scenes_may16_311am/ccvhs1do/20000 \
  /scratch/checkpoints/dronevla_v7/ccvhs1do/
```

The full GCS URI list for every may17 / may19 v7 finetune is captured
in the header of
[`pi_local_bridge/configs/moraband_v7_all.example.yaml`](../../../pi_local_bridge/configs/moraband_v7_all.example.yaml).

Verify the processor dir lands where `load_policy` expects:

```bash
ls /scratch/checkpoints/dronevla_v7/h6jtbq0w/20000/model_bfloat16/processors/
# → dronevla_v7_gate_scenes/   ← matches processor_name in the registry
```

### 6. Pick a registry YAML

Two shipped examples — pick the one that matches your host:

- [`pi_local_bridge/configs/moraband_v7_all.example.yaml`](../../../pi_local_bridge/configs/moraband_v7_all.example.yaml)
  — moraband edition. Binds to `0.0.0.0:8765`, expects checkpoints
  under `/scratch/checkpoints/dronevla_v7/`. Registers all six v7
  finetunes (history + nonhistory + four may19 variants).
- [`pi_local_bridge/configs/local_v7_all.example.yaml`](../../../pi_local_bridge/configs/local_v7_all.example.yaml)
  — falsify dev-box edition. Binds to `127.0.0.1:8765`, expects
  checkpoints under `/home/dfliu/checkpoints/dronevla_v7/`. Same
  registry shape; useful for self-hosting on the same box as the
  falsify CLI.

Both use YAML anchors (`*v7_history`, `*v7_nonhistory`, `*spec_overrides`)
so each policy entry is ~5 lines. The shape is:

```yaml
listen:
  host: 0.0.0.0
  port: 8765

auth:
  api_keys_env: PI_BRIDGE_API_KEYS    # same env var gates WS + admin

default_policy: v7-history            # which one loads at boot

policies:
  v7-history:
    <<: *v7_history                   # sequence_length=6, with_prefix
    ws_path: /v1/models/v7-history
    ckpt_path: /scratch/.../h6jtbq0w/20000/model_bfloat16
    spec_overrides: *spec_overrides

  v7-nonhistory:
    <<: *v7_nonhistory                # sequence_length absent, fixed_noise
    ws_path: /v1/models/v7-nonhistory
    ckpt_path: /scratch/.../ccvhs1do/20000/model_bfloat16
    spec_overrides: *spec_overrides
```

For a single-checkpoint deploy, the legacy
`policy:` block (`v7_history.example.yaml`, `local_v7_history.yaml`)
also works — the loader auto-wraps it into a one-entry registry.

### 7. Launch the bridge

```bash
export PI_BRIDGE_API_KEYS="pi-jt-moraband-dev-001"   # comma-list allowed
source ~/venvs/pi-local-bridge/bin/activate
python -m pi_local_bridge \
  --config pi_local_bridge/configs/moraband_v7_all.example.yaml \
  --log-level INFO
```

Successful startup sequence:

```
INFO pi_local_bridge — registry: 6 policy/policies, default=v7-history, ws_paths=[…]
INFO pi_local_bridge — patched pi_inference_client.local: …
INFO pi_local_bridge — [v7-history] loading checkpoint: /scratch/.../model_bfloat16
INFO absl — /jax/checkpoint/read/gbytes_per_sec: 2.0 GiB/s …
INFO pi_local_bridge — [v7-history] checkpoint loaded in 6.1s — action_horizon=50 action_dim=32 cameras=[…]
INFO pi_local_bridge — pi-local-bridge listening on 0.0.0.0:8765 (1 api key, 6 policies registered, active=v7-history)
```

Only the `default_policy` is loaded at boot; the other five entries are
registered but cold. They load on demand when a swap targets them.

### 8. Verify the registry + active policy

```bash
python -m pi_local_bridge.switch \
    --admin-url http://<host>:8765 \
    --api-key-env PI_BRIDGE_API_KEYS \
    --list -v
```

```
active: v7-history
loaded_at: 1779325044.7
registered:
  * v7-history                        /v1/models/v7-history
       ckpt: /scratch/.../h6jtbq0w/20000/model_bfloat16
    v7-nonhistory                     /v1/models/v7-nonhistory
       ckpt: /scratch/.../ccvhs1do/20000/model_bfloat16
    v7-nonhistory-all                 /v1/models/v7-nonhistory-all
       …
```

The `*` marks the active entry.

### 9. Smoke-infer against the active policy

In the same bridge venv (so you have `pi-inference-client` and
`websockets`):

```bash
PI_API_KEY=pi-jt-moraband-dev-001 \
  python pi_local_bridge/scripts/smoke_client.py \
    --url ws://<host>:8765/v1/models/v7-history \
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

Round-trip is ~7 s on infer #1 (JAX trace+compile) and sub-second on
infer #2+. ~50 s on infer #2 means JAX fell back to CPU — re-check
step 4.

## Adding a policy after the bridge is already up

The registry is YAML and **swaps can only target an entry that was
registered at startup** — there is no live-add endpoint. To grow it:

1. Download the new checkpoint (step 5 above; `mkdir -p` the parent
   first).
2. Add a new entry to the `policies:` block of the bridge YAML, copying
   the anchor pattern from an existing one. Pick a unique `ws_path`.
3. Restart the bridge process. Existing connections drop; the falsify
   side reconnects on next run.
4. Verify with `--list -v`. The new entry should appear, with the same
   `default_policy` still active.
5. Write a sibling falsify-side YAML under
   `configs/policies/pi_gateway/`. Copy
   [`configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml`](../../../configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml)
   and update:

   ```yaml
   gateway_url:      "ws://<host>:8765/<new_ws_path>"
   bridge_admin_url: "http://<host>:8765"
   bridge_policy_id: "<new_id>"     # must match the registry entry id
   image_size: 256                  # v7 preprocess parity — keep
   channel_order: "BGR"             # v7 preprocess parity — keep
   traceability: { … }              # at minimum wandb_run, step, checkpoint_uri
   ```

That YAML's path is what you pass to `--policy-config` on
`run_vla_episode` / `run_eval_campaign`. The CLI loads it, the policy
auto-swaps the bridge to `<new_id>` via the admin handshake, then opens
the WS.

## Swapping the active policy

Two ways, with identical effect:

**Manually, via the ops CLI** (useful for warming a policy before a
campaign, or for ad-hoc swaps):

```bash
python -m pi_local_bridge.switch \
    --admin-url http://<host>:8765 \
    --api-key-env PI_BRIDGE_API_KEYS \
    --policy-id v7-nonhistory
```

Returns once the new checkpoint is loaded (~6 s on this hardware once
the JAX backend is warm, ~30–60 s on a cold first swap). The bridge log
prints `[swap] <old> → <new> complete`.

**Automatically, via `PiGatewayPolicy`** (the normal path during a
falsify run): the policy YAML's `bridge_admin_url` + `bridge_policy_id`
fields cause the policy to GET the admin endpoint before opening the
WS. The audit line `[pi_gateway] bridge active=<id> (switch took Xs)`
lands in the CLI's stdout; `0.0s` means "already active, no-op."

In both modes the swap holds the bridge lock, closes existing WS
connections (so concurrent clients get a clean reconnect), drops the
old policy handle, runs `gc.collect() + jax.clear_caches()`, then
loads the new entry. The active policy never half-changes; either the
swap completes or the old one stays.

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

- **HTTP 409 on a WS connect** — the requested ws_path is registered
  but not the active policy. Run the swap CLI or check that the
  falsify-side YAML's `bridge_policy_id` matches the registry id.
- **HTTP 401 on admin** — `Authorization: Api-Key <key>` missing or the
  key isn't in `$PI_BRIDGE_API_KEYS`. Admin uses the same allow-list as
  the WS handshake.
- **HTTP 404 on admin switch** — `policy_id` doesn't match any
  registered entry. `GET /admin/policies` to see known ids.
- **"All connections rejected"** — `$PI_BRIDGE_API_KEYS` is unset or
  empty. Check `echo $PI_BRIDGE_API_KEYS` in the same shell launching
  the bridge.
- **`jax.errors.XlaRuntimeError: INTERNAL`** — jaxlib version mismatched
  with jax-cuda12-plugin. Reinstall with
  `pip install --upgrade "jax[cuda12]==<pinned version>"`.
- **`KeyError: 'observation/state'`** — the client used the singular
  `state=` kwarg; that always emits under `observation/joint_position`,
  which v7 doesn't accept. Use plural `states={'observation/state': vec}`
  or update the falsify-side `PiGatewayConfig.state_key` (default
  `'observation/state'` is correct).
- **`No module named 'jax'` on `--help`** — wrong venv. Activate
  `~/venvs/pi-local-bridge/` before running the bridge.
- **`gsutil … destination URL must name a directory`** — the dest dir
  parent must pre-exist for `cp -r`. `mkdir -p` it first.
- **OOM after many swaps** — JAX/XLA retains some compile + buffer
  pools across swaps even after `jax.clear_caches()`. Restart the
  bridge process; this is a known v0 limitation.

## After hosting

- **From falsify**: each `configs/policies/pi_gateway/*.yaml` carries
  `gateway_url`, `bridge_admin_url`, and `bridge_policy_id`. Run
  `falsify.cli.run_vla_episode --policy-config <yaml>` or
  `scripts/run_eval_campaign.py --policy-config <yaml>` — the policy
  handshakes the bridge automatically and writes a
  `policy_manifest.json` next to each run.
- **From a raw script**: `pi_local_bridge/scripts/smoke_client.py` is
  the reference for the minimum infer call. It does NOT perform an
  admin handshake — if you point it at a non-active ws_path you'll see
  the 409; swap first with the ops CLI.
