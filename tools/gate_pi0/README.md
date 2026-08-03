# gate-drone pi0 — serving & sim integration

End-to-end recipe for running the **gate-drone pi0** checkpoints (the
`gate-drone-pi0-bucket`) inside falsify's sim, including the **pin**
(source-noise) research variant. This is the version-controlled companion to
the ephemeral `local/` bucket download — everything here survives a
`local/` wipe.

These are stock **openpi pi0** checkpoints (PaliGemma, H=50, action_dim=32,
7-D control state/action on x,y,z,yaw), *not* Pi-gateway checkpoints — so
`pi_local_bridge` / `PiGatewayPolicy` cannot serve them. They load via
`openpi` and are served over openpi's websocket protocol, which falsify's
`VLAPolicy` speaks natively.

---

## 1. Downloads

| Thing | Location | How |
|---|---|---|
| Checkpoints + assets + `gate_inference.py` | HF bucket `hf://buckets/denis-liu-tri/gate-drone-pi0-bucket` (~18 GB) | `hf sync` (needs `huggingface_hub>=1.20`; the `hf sync` subcommand + `hf://buckets/` scheme are 1.x-only) |
| openpi source | `https://github.com/Physical-Intelligence/openpi.git` @ `16affa3` | `git clone` + `git checkout 16affa3` |

```bash
# Bucket (isolated hf 1.x via uvx so the global huggingface_hub 0.36 stays put):
uvx --from 'huggingface_hub>=1.20' hf sync \
    hf://buckets/denis-liu-tri/gate-drone-pi0-bucket ./local
```

Checkpoints in the bucket:
- `gate_both_scratch` — synth + real, standard pi0. **Recommended baseline.**
- `gate_both_pin` — synth + real, **pin** variant (needs `assets/pin_U_gate_k5.npy` + `assets/prior_gate_mlp.pt`).
- `gate_synth_scratch` — synth only.

Deployment contract (bucket README): **RGB**, two cams
(`observation/image` fwd + `observation/wrist_image` down), 224², 10 fps,
7-D EE-delta actions, **replan every ~8 steps** (RecedingHorizon).

---

## 2. openpi setup (isolated venv)

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ~/code/openpi
git -C ~/code/openpi checkout 16affa3          # the validated SHA
cd ~/code/openpi && env -u VIRTUAL_ENV uv sync # own .venv, JAX cuda12 — NOT the falsify/SousVide venv
```

Always run openpi-side commands with
`env -u VIRTUAL_ENV ~/code/openpi/.venv/bin/python …` (the shell exports a
stale `VIRTUAL_ENV` pointing at the SousVide venv).

### 2a. Apply the openpi patch

Upstream openpi at `16affa3` is missing two things this checkpoint family
needs. Both are in **`openpi_gate_pi0.patch`** (this dir):

1. **`pi0_gate` config** (`training/config.py`) — a local edit on the
   training box that never got committed. Reconstructed from `pi0_libero`
   (the LIBERO `LiberoInputs`/`Outputs` transforms already match
   `gate_inference.py`'s fed keys: `observation/image`,
   `observation/wrist_image`, `observation/state`, `prompt`) repointed at
   `repo_id="local/gate_nav"`. The real base was a custom `pi0_libero_shared`,
   but its delta is training-only (norm-stats sharing) and irrelevant to
   inference.
2. **Source-noise ("pin") threading** (`policies/policy.py` +
   `models/pi0.py`) — lets `Policy.infer(obs, noise=…)` inject the flow's
   initial `x_1` instead of sampling it. Required by the pin path.

```bash
git -C ~/code/openpi apply /path/to/falsify/tools/gate_pi0/openpi_gate_pi0.patch
```

> **On the training machine** the real `pi0_gate` (via
> `source-noise-mvp/experiments/rung3/patch_gate_config.py`) and the
> source-noise fork already exist — skip the patch there and just use the
> serve scripts + the falsify-side `--action-space` flag.

### 2b. The norm-stats gotcha (handled in the serve scripts)

The shipped norm stats are **7-D**, but openpi normalizes at the padded
model dim (**32**): `LiberoInputs` pads state→32, then `Normalize`; on the
output side model actions (32) → `Unnormalize` → `LiberoOutputs` slices to 7.
Feeding 7-D stats → broadcast error. The serve/smoke scripts pad state+actions
stats to 32 (mean 0 / std 1 on dims 7-31) — faithful because those dims are
always the zero-padding. **`gate_inference.py`'s `GatePolicy` is broken
as-shipped without this.**

---

## 3. Serve

Launch with `XLA_PYTHON_CLIENT_PREALLOCATE=false` — otherwise JAX grabs ~75 %
(~19 GB) of the GPU and starves falsify's gsplat renderer (both share one GPU).

**Baseline (`gate_both_scratch`):**
```bash
env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false ~/code/openpi/.venv/bin/python \
    tools/gate_pi0/serve_gate.py \
    --ckpt local/checkpoints/gate_both_scratch --norm local/assets/gate_nav \
    --host 127.0.0.1 --port 8000 \
    --default-prompt "go through the gate on the left and hover over the stuffed animal"
```

**Pin (`gate_both_pin`):** computes `c = MLP([state, left/right onehot])` and
injects `noise = g − (g·U)Uᵀ + (c·Uᵀ)` per query.
```bash
env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false ~/code/openpi/.venv/bin/python \
    tools/gate_pi0/serve_gate_pin.py \
    --ckpt local/checkpoints/gate_both_pin --norm local/assets/gate_nav \
    --pin-u local/assets/pin_U_gate_k5.npy --prior local/assets/prior_gate_mlp.pt \
    --host 127.0.0.1 --port 8000 \
    --default-prompt "go through the gate on the left and hover over the stuffed animal"
```

`smoke_gate.py` is a no-server standalone check that the checkpoint loads and
emits a sane `(50,7)` chunk.

---

## 4. Run in the sim (falsify side)

`run_vla_episode` with **no** `--policy-config` builds a default `VLAPolicy`
from `--host/--port/--image-size`. Prompts must match training verbatim (use
`--prompt-name left_gate` / `right_gate`).

```bash
source tools/env.sh
export PYTHONPATH=src:external/FiGS/src:external/splatnav:$PYTHONPATH
.venv/bin/python -m falsify.cli.run_vla_episode \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --prompt-name left_gate \
    --host 127.0.0.1 --port 8000 \
    --image-size 224 \
    --action-space absolute \      # REQUIRED for these checkpoints (see below)
    --actions-per-chunk 8 \        # receding horizon per the README (replan ~8, not 50)
    --horizon-s 60 \
    --out runs/gate_pi0_smoke/left_gate_pin
```

### `--action-space absolute` (why)

These pi0 checkpoints emit **absolute MOCAP position/yaw waypoints** (the
delta-action model adds the current state back server-side), *not* per-step
deltas. The default `VLAPolicy` `action_space="delta"` cumsum-integrates them
→ geometric blow-up (z 1.5 → 78 → 3983 …). Tell-tale: action col 2 (z) ≈ 1.53
= absolute altitude; action norm stats are delta-like (mean≈0, std≈0.004)
because the training deltas are chunk-relative. `absolute` mode also
**re-anchors** each chunk to the drone's current pose (keeps the model's
relative motion shape, removes the teleport-replay seam discontinuity).

---

## 5. Findings (what we tried)

- **Integration is correct & glitch-free.** Full path validated:
  bucket → openpi (`pi0_gate` + norm padding + source-noise patch) → serve →
  `VLAPolicy` (absolute + re-anchor) → sim rollout.
- **Pin works and measurably shifts the trajectory** vs scratch — the
  instruction-prior pinning has a real, visible effect in-sim.
- **Chunk-seam "teleport" glitch** (raw absolute + v0 teleport-replay
  integrator): ~2-3 cm backward hitch at every re-query seam. Fixed by
  re-anchoring absolute chunks to the current pose (`VLAPolicy`, 0/74 seams).
- **The apparent forward flight was mostly that glitch.** Raw-absolute moved
  1.33 m/60 s at replan-8, but ~1.37 cm/chunk of that came from seam teleports
  vs the model's genuine ~0.5 cm/chunk. With honest (re-anchored) integration
  the policy barely moves.
- **Real bottleneck = domain / start-pose gap.** The model emits targets ~2 cm
  *behind* itself with ~5 mm forward span → timid → hovers. Training state
  mean x≈1.17 but the drone starts at x≈0 (out of distribution). Next lever:
  start the drone in-distribution and/or improve render fidelity vs training
  footage — not more integration tweaks.

---

## Files in this dir

| File | Runs in | Purpose |
|---|---|---|
| `openpi_gate_pi0.patch` | applied to `~/code/openpi` | pi0_gate config + source-noise threading |
| `smoke_gate.py` | openpi venv | standalone checkpoint load + one inference |
| `serve_gate.py` | openpi venv | websocket server (baseline, padded norm stats) |
| `serve_gate_pin.py` | openpi venv | websocket server (pin: prior MLP + U-subspace noise) |

`gate_inference.py` (the upstream serving module) and `export_pin_prior.py`
ship in the bucket itself; note `gate_inference.py` needs the norm-stats
padding fix to run.
