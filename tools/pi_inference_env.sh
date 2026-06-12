# Pi inference-client env shim.
#
# Two distinct credentials are involved when running the pi07 dronevla
# finetune through the policy gateway:
#
# 1. Pulling the `pi-inference-client` wheel from Pi's Artifact Registry.
#    Uses a GCP service-account key — the SAME one the dataset validator
#    uses for the partner GCS bucket (`pi-external-partners-…json`).
#    Sourced when running `uv pip install` against the private index.
#
# 2. Connecting to a deployed gateway at runtime
#    (`wss://api.pi-fleet.com/v1/models/<model_id>`). Uses a route-scoped
#    Pi API key (string starting with `pi-…`) obtained from your Pi
#    contact. **Not** a GCP token. Exported here as $PI_API_KEY so the
#    policy YAMLs can reference `${env:PI_API_KEY}` without checking the
#    secret into the repo.
#
# Source this file from the repo root *in addition to* tools/env.sh:
#
#   source tools/env.sh                    # gcc-11 + PYTHONPATH (CUDA JIT)
#   source tools/pi_inference_env.sh       # GCP token + PI_API_KEY
#
# Re-source whenever the GCP access token expires (~1h).
#
# ---------------------------------------------------------------------------
# 1. GCP access token for `uv pip install pi-inference-client`.
# ---------------------------------------------------------------------------

_PI_SA_KEY="${PI_INFERENCE_SA_KEY:-$HOME/code/dataset_validation/pi-data-sharing/pi-external-partners-814af4af5a99.json}"
_PI_SA_EMAIL="dronevla-external-sa@pi-external-partners.iam.gserviceaccount.com"

if [ ! -f "$_PI_SA_KEY" ]; then
  echo "[pi_inference_env] WARN: SA key not found at $_PI_SA_KEY" >&2
else
  gcloud auth activate-service-account --key-file="$_PI_SA_KEY" >/dev/null 2>&1
  gcloud config set account "$_PI_SA_EMAIL" >/dev/null 2>&1
  PI_GCP_ACCESS_TOKEN="$(gcloud auth print-access-token 2>/dev/null)"
  if [ -n "$PI_GCP_ACCESS_TOKEN" ]; then
    export PI_GCP_ACCESS_TOKEN
    export PI_PYTHON_INDEX_URL="https://oauth2accesstoken:${PI_GCP_ACCESS_TOKEN}@us-east5-python.pkg.dev/pi-external-partners/pi-python/simple/"
    # Convenience: ready-to-paste install command.
    : "${PI_INSTALL_HINT:=uv pip install --extra-index-url \"\$PI_PYTHON_INDEX_URL\" pi-inference-client}"
    export PI_INSTALL_HINT
  else
    echo "[pi_inference_env] WARN: failed to mint GCP access token for $_PI_SA_EMAIL" >&2
  fi
fi

# ---------------------------------------------------------------------------
# 2. Pi gateway API key (runtime credential).
# ---------------------------------------------------------------------------
# Lives in tools/secrets.env (gitignored), sourced here so it survives shell
# restarts. Policy YAMLs reference it via `${env:PI_API_KEY}`; falsify-gui
# reads the same file at startup.

_FALSIFY_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
if [ -f "$_FALSIFY_TOOLS_DIR/secrets.env" ]; then
  . "$_FALSIFY_TOOLS_DIR/secrets.env"
fi
unset _FALSIFY_TOOLS_DIR

if [ -z "${PI_API_KEY:-}" ]; then
  echo "[pi_inference_env] note: PI_API_KEY is unset — gateway connections will fail." >&2
  echo "[pi_inference_env] export PI_API_KEY=pi-… (route-scoped key from your Pi contact)" >&2
fi

# ---------------------------------------------------------------------------
# 3. JAX <-> gsplat GPU-share defaults.
# ---------------------------------------------------------------------------
# When the pi_local_bridge (JAX) and the falsify rollout (gsplat / torch)
# share a GPU on the same host, JAX's default 75% preallocation steals all
# the memory before falsify's renderer can load. These two exports make JAX
# allocate on-demand and cap its share, so gsplat has headroom.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
