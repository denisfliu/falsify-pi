# pi-local-bridge

Self-hosted WebSocket bridge that **speaks pi-inference-client's gateway
protocol** against a **locally-loaded** Pi checkpoint.

Why this exists: Pi's commercial gateway is the easiest way to query a
deployed checkpoint, but it requires Pi to host the inference. When we
want to run a checkpoint on our own hardware (moraband, etc.) while still
using `pi_inference_client.PolicyClient` on the falsify side unchanged,
this bridge fills the gap — it wraps `pi_inference_client.local` (JAX
in-process inference) behind a WSS endpoint that re-emits the same wire
protocol Pi's gateway publishes.

## Architecture

```
falsify (this repo)                              moraband (or other GPU host)
─────────────────────────                        ────────────────────────────
PiGatewayPolicy                                  pi-local-bridge
  └─ pi_inference_client.PolicyClient            ├─ websockets.sync.server
        │   wss://moraband:8765/v1/models/…      │     ↑
        └────────── Api-Key ─────────────────────┤     │
                                                 ├─ msgpack-framed dispatch
                                                 │     ↓
                                                 └─ pi_inference_client.local
                                                     ├─ load_policy (JAX/Flax)
                                                     └─ LocalPolicyClient.infer
```

Every Pi-client send is `msgpack_numpy.packb((api, payload))`; the bridge
unpacks, dispatches into a `LocalPolicyClient`, and re-packs the result.

## Install (on the GPU host)

The bridge depends on `pi-inference-client[local]`, which lives in Pi's
private Artifact Registry. Use the partner SA key (the one falsify also
uses for the GCS dataset bucket) to pull it:

```bash
gcloud auth activate-service-account --key-file=~/pi-external-partners-...json
gcloud config set account dronevla-external-sa@pi-external-partners.iam.gserviceaccount.com
TOKEN=$(gcloud auth print-access-token)

# Use a *separate* venv from falsify — JAX 0.5+ / flax / orbax don't
# share well with the nerfstudio + gsplat + cu121 stack.
python -m venv ~/venvs/pi-local-bridge
source ~/venvs/pi-local-bridge/bin/activate

pip install --extra-index-url \
  "https://oauth2accesstoken:${TOKEN}@us-east5-python.pkg.dev/pi-external-partners/pi-python/simple/" \
  -e /path/to/falsify/pi_local_bridge
```

## Get a checkpoint onto the box

```bash
gsutil -m cp -r \
  gs://dronevla-raw-data/model_checkpoints/2026_may17/mar16_history_ft_dronevla_v7_gate_scenes_may16_311am/h6jtbq0w/20000 \
  /scratch/checkpoints/dronevla_v7/h6jtbq0w/
```

The processor directory (`processors/dronevla_v7_gate_scenes/`) needs to
sit *under* the model directory referenced by `ckpt_path`, matching
`pi_inference_client.local.load_policy`'s search rule
(`ckpt_path/processors/<processor_name>`). Verify with `ls -R` after the
copy.

## Run

```bash
export PI_BRIDGE_API_KEYS="pi-some-route-key,pi-another-route-key"
python -m pi_local_bridge --config configs/v7_history.example.yaml
```

First request after startup pays the JAX trace+compile cost (~30–60 s);
subsequent calls run at the steady-state inference rate.

## Multi-policy hosting

The bridge can hold a **registry** of N policy entries (each with its own
`ws_path`) and rotate which one is GPU-resident on demand. Exactly one is
active at a time. See [`configs/moraband_v7_all.example.yaml`](configs/moraband_v7_all.example.yaml)
for the registry shape — it bundles every downloaded dronevla v7 finetune
behind one bridge instance on port 8765.

**No implicit swap.** A client opening a WS to a registered `ws_path`
whose policy isn't the active one is **rejected with HTTP 409**:

```
$ curl -i ws://moraband:8765/v1/models/v7-nonhistory-center
HTTP/1.1 409 Conflict
policy_id='v7-nonhistory-center' is registered but not active;
current active='v7-history'. GET /admin/switch_policy?policy_id=v7-nonhistory-center to switch.
```

This guarantees a run can never accidentally consume the wrong checkpoint.

**Admin endpoints** (same port as the WS server, same `Authorization: Api-Key`
header as the WS handshake):

| Method | Path                              | Purpose                                       |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/admin/policies`                 | list registry + active id + per-entry ckpt_path |
| GET    | `/admin/switch_policy?policy_id=X`| evict the current policy, load X synchronously |

(GET-only because the `websockets` sync server's HTTP parser rejects
non-GET methods before our `process_request` hook runs. Idempotent +
internal-only, so this is fine.)

Switching is synchronous and blocks while the new checkpoint loads
(~30–60 s on first swap). All existing WS connections are closed before
the swap so clients can't observe a torn-down policy.

**Ops CLI** for one-off swaps from anywhere on the network:

```bash
# List
python -m pi_local_bridge.switch \
    --admin-url http://moraband.stanford.edu:8765 \
    --api-key-env PI_BRIDGE_API_KEYS --list

# Switch
python -m pi_local_bridge.switch \
    --admin-url http://moraband.stanford.edu:8765 \
    --api-key-env PI_BRIDGE_API_KEYS \
    --policy-id v7-nonhistory-center
```

**Falsify-side handshake.** `PiGatewayPolicy` calls `/admin/switch_policy`
automatically before opening its WS when the policy YAML sets
`bridge_admin_url` + `bridge_policy_id`. The CLI emits an audit line and
writes a `policy_manifest.json` next to each run capturing the YAML
sha256, the bridge URL, the bridge_policy_id, and the post-swap
`active_policy_id` reported by the bridge — so a reviewer can prove
which checkpoint produced any artifact without trusting filenames.

### Memory caveat across swaps

JAX/XLA retains some compile + buffer pools after `del policy +
jax.clear_caches()`. After many swaps the resident set can creep up; if
it drives OOM, restart the bridge process. This is a known limitation —
the alternative (out-of-process per-policy workers) was deferred in v0.

### Adding a new policy to the registry

There is no live-add endpoint — the registry is YAML and a swap can only
target an entry that was registered at startup. To grow it:

1. **Download the checkpoint** to a stable path (the dest dir parent must
   pre-exist or `gsutil -m cp -r` fails):

   ```bash
   mkdir -p /scratch/checkpoints/dronevla_v7/<wandb_run>
   gsutil -m cp -r \
     gs://dronevla-raw-data/model_checkpoints/.../<wandb_run>/<step> \
     /scratch/checkpoints/dronevla_v7/<wandb_run>/
   ls /scratch/checkpoints/dronevla_v7/<wandb_run>/<step>/model_bfloat16/processors/
   # → dronevla_v7_gate_scenes/   ← matches processor_name in the registry
   ```

2. **Append an entry** to the bridge's `policies:` block (use a YAML
   anchor — every v7 finetune shares the same processor + wire layout):

   ```yaml
   v7-nonhistory-mynew:
     <<: *v7_nonhistory                # or *v7_history for sequence_length=6
     ws_path: /v1/models/v7-nonhistory-mynew
     ckpt_path: /scratch/checkpoints/dronevla_v7/<wandb_run>/<step>/model_bfloat16
     spec_overrides: *spec_overrides
   ```

3. **Restart the bridge.** Adding entries doesn't require a `default_policy`
   change — leave it at whichever id you typically boot active.

4. **Verify** the new entry is registered without forcing a swap yet:

   ```bash
   python -m pi_local_bridge.switch \
       --admin-url http://<host>:8765 \
       --api-key-env PI_BRIDGE_API_KEYS --list -v
   ```

5. **Add a sibling falsify-side YAML** under
   `configs/policies/pi_gateway/` so the CLI can pick the new policy by
   name. Required fields are `gateway_url` (with the matching ws_path),
   `bridge_admin_url`, `bridge_policy_id`, `api_key`, plus the v7
   preprocess parity knobs (`image_size: 256`, `channel_order: "BGR"`)
   and the camera_map. Copy
   [`configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml`](../configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml)
   as a starting point.

### Backwards compatibility

The legacy single-`policy:` block (used by `v7_history.example.yaml` and
`local_v7_history.yaml`) still works — it auto-wraps into a one-entry
registry. No migration is needed for single-checkpoint deployments.

## Connect from falsify

On the falsify host, the `PiGatewayPolicy` YAML names both the WS path
and the admin endpoint so the policy can perform the swap automatically:

```yaml
# configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml
gateway_url:       "ws://moraband.stanford.edu:8765/v1/models/v7-history"
api_key:           "${env:PI_API_KEY}"   # must match one of PI_BRIDGE_API_KEYS

# Bridge admin handshake — PiGatewayPolicy._ensure_connected GETs
# /admin/switch_policy?policy_id=<bridge_policy_id> on this URL before
# opening the WS. The same Api-Key is reused for the admin call.
bridge_admin_url:  "http://moraband.stanford.edu:8765"
bridge_policy_id:  "v7-history"   # must match the registry entry id
```

Source `tools/pi_inference_env.sh` (in the falsify repo) to populate
`$PI_API_KEY`, then run any falsify CLI that uses `PiGatewayPolicy`. The
CLI writes a `policy_manifest.json` next to each run capturing the YAML
sha256 and the bridge handshake response so a reviewer can prove which
checkpoint produced the artifacts.

## TLS

For in-network use over the Stanford VPN, plain `ws://` is fine — the
client logs a warning but allows it. For exposure beyond the trusted
network, terminate TLS upstream (nginx / traefik / Cloudflare tunnel)
and front the bridge with `wss://`.

## Upstream patches applied at startup

`server._apply_local_api_patches()` monkey-patches two functions in
`pi_inference_client.local.policy_api` before `load_policy` runs. Both are
required to make the dronevla v7 gate-scenes checkpoints load:

1. **`_resolve_image_key` → identity passthrough.** Upstream assumes
   payload keys are `observation/<camera>/rgb/image` with a non-empty
   camera segment. The v7 base camera is just `observation/rgb/image` —
   no camera name — and the heuristic can't construct that. The patch
   short-circuits when `user_key in valid_fields`, letting the bridge
   YAML name server-side keys directly via `image_keys`.

2. **`_extract_process_input_fields` → history-aware shape parse.**
   The upstream regex grabs the first three numbers in an image input's
   shape and calls them `(image_h, image_w, channels)`. For v7 with
   `sequence_length=6`, the shape is `(6, 448, 448, 3)` and the parser
   ends up with `image_h=6, image_w=448`. The replacement walks the
   inner tuple, finds the trailing `3` channel dim, and returns the
   two preceding values as `(H, W)`.

Both patches are additive (no behaviour change for already-supported
checkpoints) and idempotent. If upstream fixes either case, the patch
becomes a no-op.

## State key wiring (client → bridge)

The Pi client's default `InputData(state=…)` puts the vector under
`observation/joint_position`. The v7 server expects it under
`observation/state`. Use the plural `states={"observation/state": vec}`
form on the client side instead — `PiGatewayPolicy` does this
automatically; if you write a raw client (e.g. `scripts/smoke_client.py`)
you have to do it yourself.

## What's *not* implemented

- **Video-encoded image payloads** (`encode_as_video: true`). The bridge
  expects JPEG bytes (the default) or raw ndarray. Client side: keep
  `video_encode: false` in `ClientConfig`.
- **Multiple concurrent inferences.** A single `RLock` serializes all
  `infer` calls — JAX is single-stream anyway. Multiple clients connect
  fine; their requests interleave.
- **Per-connection history isolation.** History-mode buffer is reset on
  every new connection and on every `reset` message. Don't run two
  simultaneous episodes against a history-mode endpoint.
- **WSS redirects** the client supports. We're a leaf endpoint, not a
  load balancer.
