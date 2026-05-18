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

## Connect from falsify

On the falsify host, point the existing `PiGatewayPolicy` YAML at the
bridge:

```yaml
# configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml
gateway_url: "ws://moraband.stanford.edu:8765/v1/models/v7-history"
api_key:     "${env:PI_API_KEY}"   # must match one of PI_BRIDGE_API_KEYS
```

Source `tools/pi_inference_env.sh` (in the falsify repo) to populate
`$PI_API_KEY`, then run any falsify CLI that uses `PiGatewayPolicy`.

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
