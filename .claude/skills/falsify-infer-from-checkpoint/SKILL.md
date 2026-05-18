---
name: falsify-infer-from-checkpoint
description: Query a Pi dronevla v7 checkpoint from the falsify side via `PiGatewayPolicy`. Works against either a self-hosted `pi_local_bridge` (see falsify-host-checkpoint) or a Pi-hosted gateway URL. Covers the smoke-client validation path, the `run_vla_episode --policy-config` full-rollout path, and the YAML knobs that decouple checkpoint metadata from rollout invocation.
---

# falsify-infer-from-checkpoint

The client side of the pi-inference-client gateway protocol. Companion
to [`falsify-host-checkpoint`](../falsify-host-checkpoint/SKILL.md),
which sets up the *server*. Use this skill once a gateway URL exists —
either a Pi-hosted `wss://api.pi-fleet.com/...` route or a self-hosted
`ws://moraband:8765/...` bridge.

## When to use

- We have a `PiGatewayPolicy` YAML pointing at a running gateway and
  want to verify connectivity / shape correctness before a real rollout.
- We're swapping in a new v7 finetune (history ↔ nonhistory, different
  W&B run, different step) and want to do a one-line YAML change rather
  than touching Python code.
- A rollout is misbehaving and we want to isolate the policy-server
  contract from the rest of the falsify stack.

Not for: rollouts that talk to the older OpenPI server protocol (use
the legacy [`falsify-trajectory-from-vla`](../falsify-trajectory-from-vla/SKILL.md) — that path is `VLAPolicy`, not
`PiGatewayPolicy`).

## Inputs

- A gateway URL — either Pi-hosted (`wss://api.pi-fleet.com/v1/models/<id>`)
  or self-hosted (`ws://host:port/v1/models/<name>` from
  `falsify-host-checkpoint`).
- An API key string the gateway accepts. For Pi-hosted, it's the route
  key Pi gave you; for self-hosted, anything in `$PI_BRIDGE_API_KEYS`.
- A scene + frame YAML (only for the full-rollout path through
  `run_vla_episode`).

## YAML organization

Each deployed checkpoint gets one falsify-side YAML under
`configs/policies/pi_gateway/`. The two shipped variants:

| File | Variant | Default gateway URL |
|---|---|---|
| `history_h6jtbq0w_20k.yaml` | pi07 history (best MSE) | `ws://moraband:8765/v1/models/v7-history` |
| `nonhistory_ccvhs1do_20k.yaml` | pi07 nonhistory | `ws://moraband:8766/v1/models/v7-nonhistory` |

Both reference `${env:PI_API_KEY}` so the key never lands in git. Source
`tools/pi_inference_env.sh` (or just `export PI_API_KEY=…`) before any
falsify CLI that loads the YAML.

To target a **locally-running bridge** on this machine, copy one of
the YAMLs (or override the field at load time):

```yaml
gateway_url: "ws://127.0.0.1:8765/v1/models/v7-history"
api_key:     "${env:PI_API_KEY}"
```

To target **Pi-hosted** once Pi deploys the checkpoint:

```yaml
gateway_url: "wss://api.pi-fleet.com/v1/models/<id-pi-gives-you>"
```

No other field changes; the entire wire protocol is gateway-URL-agnostic.

## Procedure — sanity check (no scene needed)

The fastest way to verify a gateway is reachable, authenticated, and
returning correctly-shaped actions:

```bash
source ~/venvs/pi-local-bridge/bin/activate   # OR the falsify venv since 0.4.9
export PI_API_KEY=pi-some-allowed-key
python pi_local_bridge/scripts/smoke_client.py \
  --url ws://127.0.0.1:8765/v1/models/v7-history \
  --n-iters 2
```

Expected (on a CUDA GPU):

```
connected + load handshake in 0.01s
server cameras:    ['observation/rgb/image', 'observation/wrist_image/rgb/image']
action_horizon:    50
action_dim:        32
image_resolution:  (448, 448)
infer #1  client.infer round-trip: 0.4s  actions.shape: (25, 7)  raw_actions.shape: (50, 32)
infer #2  client.infer round-trip: 0.2s  actions.shape: (25, 7)
reset OK
smoke OK
```

If round-trip is ~50s, JAX on the server fell back to CPU — see
`falsify-host-checkpoint` step 4 (install `jax[cuda12]`).

Why `action_dim=32` but `actions.shape=(25, 7)`: the model's native
action space is 32-D (pi07 backbone, originally for manipulator). The
processor's `unprocess` step projects it back to the 7-D drone action
the falsify policy expects. `raw_actions` is the model-space 32-D
output exposed for debugging.

## Procedure — full episode rollout

`run_vla_episode` accepts `--policy-config` to dispatch to
`PiGatewayPolicy` instead of constructing a `VLAPolicy` from
`--host/--port/--image-size`:

```bash
source /home/dfliu/code/falsify/.venv/bin/activate
source tools/env.sh                     # gcc-11 + PYTHONPATH for gsplat
export PI_API_KEY=pi-some-allowed-key   # must match the gateway allow-list

PYTHONPATH=src:external/FiGS/src:external/splatnav \
python -m falsify.cli.run_vla_episode \
  --scene configs/scenes/left_gate.yaml \
  --frame configs/frames/carl_dual.yaml \
  --prompt "fly through the gate" \
  --policy-config configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml \
  --horizon-s 30 \
  --out runs/v7_history_smoke_$(date +%Y%m%d_%H%M)
```

When `--policy-config` is set, the CLI:
- Skips the openpi smoke handshake (the YAML's gateway is WSS, not
  raw moraband:8000).
- Smoke-imports `pi_inference_client` instead of `openpi_client`.
- Pulls `hz` and `execute_chunk_size` from the YAML (the `--hz` /
  `--actions-per-chunk` flags are ignored for `pi_gateway`).
- Writes the YAML's `traceability:` block into
  `runs/<stamp>/episode_summary.json` so the run is reproducible.

## Procedure — orchestrator smoke

`smoke_test.py` also dispatches `type: pi_gateway`:

```yaml
# configs/falsification/smoke_pi_gateway.yaml
scene:  configs/scenes/left_gate.yaml
frame:  configs/frames/carl_dual.yaml
policy: configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml
horizon_s: 10
hz: 30
```

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.smoke_test \
  --config configs/falsification/smoke_pi_gateway.yaml \
  --stub-recovery
```

Useful for verifying the policy + sensor-rig + frame-graph integration
without engaging the splatnav GPU path.

## Frame contract reminder

`PiGatewayPolicy` does the NED↔MOCAP boundary identically to the legacy
`VLAPolicy` (perm5, negated yaw out). The v7 finetune was trained on
the same dronevla MOCAP-z-up convention, so the bridge sees:

- State sent: `(7,) float32 = [px_mocap, py_mocap, pz_mocap, -yaw_ned, 0, 0, 0]`
- Actions returned: `(chunk, 7)` after unprocessing, where cols 0..2 are
  MOCAP position deltas, col 3 is MOCAP yaw delta, cols 4..6 are
  constant zero by training distribution.
- Integration: cumsum positions in MOCAP, convert chunk back to NED,
  attach NED-frame quaternions.

If a new checkpoint changes the action layout (e.g., velocity-targets
instead of position-deltas, or different yaw column), edit
`action_pos_slice` and `action_yaw_index` in the policy YAML — no
Python code changes.

## YAML knobs that decouple checkpoint metadata from invocation

All of these live in `configs/policies/pi_gateway/<variant>.yaml`:

| Field | Purpose |
|---|---|
| `gateway_url` | The only thing that changes when switching Pi-hosted ↔ self-hosted ↔ a different checkpoint URL. |
| `api_key` | `${env:PI_API_KEY}`; never bake the key into git. |
| `execute_chunk_size` | Actions delivered per request; `run_vla_episode`'s `--actions-per-chunk` is ignored in favor of this. |
| `prompt` | Task instruction; the CLI's `--prompt` is still passed but the YAML wins when both are set. |
| `hz` | Control rate; `run_vla_episode --hz` is ignored. v7 = 30. |
| `state_dim` / `action_dim` | Width of state/action vectors. v7 = 7/7. |
| `action_pos_slice` / `action_yaw_index` | Which action columns are NED-pos-deltas / yaw-delta. v7 = `[0,3]` / `3`. |
| `camera_map` | `{forward, downward} → server payload key`. v7 maps to `observation/rgb/image` and `observation/wrist_image/rgb/image`. |
| `state_key` | Server-side state input name. v7 = `observation/state` (the singular Pi default `observation/joint_position` doesn't match). |
| `server_frame` | Frame the policy was trained in. v7 = `mocap`. |
| `traceability` | Free-form metadata (W&B run, step, GCS URI, processor_name). Not load-bearing; written next to debug bundles. |
| `record_dir` | If set, each query saves a `query_NNNN_step_KKKKK/` bundle (native renders, raw actions, integrated NED waypoints, metrics). |

## Troubleshooting

- **`ServerConnectionError: Not connected` / `401 Unauthorized`** —
  `$PI_API_KEY` not in the gateway's allow-list. For self-hosted,
  echo `$PI_BRIDGE_API_KEYS` on the server host.
- **`gateway_url does not use wss://`** — info-only warning from
  `pi_inference_client.PolicyClient.connect()`. Plain `ws://` is fine
  on a trusted in-network bridge.
- **`InferenceError: KeyError: 'observation/state'`** — the client is
  using the singular `state=` kwarg; that always routes to
  `observation/joint_position`. The fix is in `PiGatewayPolicy` already
  (uses `states={state_key: vec}`). If reproducing in a raw client,
  use plural `states=` with the explicit key.
- **`InferenceError: ValueError: Error concatenating attention info`**
  — the bridge is loading a v7 history checkpoint without the
  `_extract_process_input_fields` shape patch. Confirm
  `_apply_local_api_patches` is called (you should see
  `patched pi_inference_client.local: …` in the bridge log on startup).
- **Round-trip ≫ 1s on GPU host** — model trace+compile is happening on
  the first call. Successive calls amortize. If they all stay slow,
  jax fell back to CPU; check `jax.default_backend()` on the host.
- **`module 'falsify.policy' has no attribute 'PiGatewayPolicy'`** —
  `pi-inference-client` not installed in the falsify venv. From repo
  root: `uv pip install --no-deps pi_local_bridge/wheels/pi_inference_client-*.whl`
  or (with auth) `uv pip install --extra-index-url <oauth2 url> pi-inference-client`.
- **`websockets.sync.client` import error** — falsify venv has
  websockets <12. `uv pip install "websockets>=12,<17"`.

## Where the work happens

| Component | File | Notes |
|---|---|---|
| Client class | `src/falsify/policy/pi_gateway.py` | `PiGatewayPolicy.observe` + the NED↔MOCAP boundary. |
| Config dataclass | same | `PiGatewayConfig` — every YAML knob above lives here. |
| Variant YAMLs | `configs/policies/pi_gateway/*.yaml` | One per deployed checkpoint. |
| Smoke runner | `pi_local_bridge/scripts/smoke_client.py` | No falsify import; uses `pi_inference_client.PolicyClient` directly. |
| Dispatch (orchestrator smoke) | `src/falsify/cli/smoke_test.py::_policy_factory_from_yaml` | `kind == "pi_gateway"` branch. |
| Dispatch (full rollout) | `src/falsify/cli/run_vla_episode.py` | `--policy-config` arg. |
| Env helper | `tools/pi_inference_env.sh` | Activates SA + reminds about `$PI_API_KEY`. |
