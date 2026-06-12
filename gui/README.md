# falsify-gui

Local web GUI for managing the falsify repo: launching and monitoring jobs
(eval campaigns, recovery collection, VLA episodes, training-data export,
trajectory planning), browsing `runs/` artifacts, switching the active
checkpoint on the pi_local_bridge, and driving the scene/gsplat viz tools.

## Design in one paragraph

The backend (FastAPI) **never imports falsify**. Every workflow runs as a
subprocess of the repo-root venv's python under `tools/env.sh`
(`bash -c 'source tools/env.sh && exec "$@"' -- .venv/bin/python …`), with
stdout+stderr captured to a per-job log file under `gui/data/jobs/`. Job
state lives in sqlite (`gui/data/jobs.db`); the browser tails logs over SSE.
Jobs run in their own process groups and **survive GUI restarts** — on
startup the server re-adopts still-running pids (guarded against pid reuse
by `/proc` starttime). Plotly HTMLs produced by the repo's own viz tools are
embedded via iframe; quick trajectory-NPZ plots are rendered on demand with
the GUI venv's numpy+plotly (torch is deliberately not a dependency here).

## Setup

```bash
cd gui
uv venv --python 3.11
uv pip install -p .venv/bin/python -e .   # -p matters: your shell's
                                          # $VIRTUAL_ENV points at the shared
                                          # SousVide venv — do NOT install there
```

## Run

```bash
gui/.venv/bin/python -m falsify_gui              # http://127.0.0.1:9000
```

Secrets are auto-loaded from **`tools/secrets.env`** (gitignored, `export
KEY=value` lines) at startup; explicit environment variables win over the
file. The relevant keys:

- `PI_API_KEY` — inherited by launched jobs; policy YAMLs resolve
  `${env:PI_API_KEY}` for gateway connections. Without it, pi_gateway jobs
  fail their handshake.
- `PI_BRIDGE_API_KEYS` — used by the Bridge tab's admin proxy
  (`Authorization: Api-Key <first entry>`); falls back to `PI_API_KEY`.
- `FALSIFY_GUI_BRIDGE_URL` — override the bridge admin URL (default: the
  most common `bridge_admin_url` across `configs/policies/pi_gateway/`).

The `/api/health` endpoint (and the header badge) reports whether the keys
are present in the server env.

## Security

There is **no auth**. The server binds `127.0.0.1` by default; passing
`--host 0.0.0.0` exposes job-launching-as-you, the kill endpoints, and all
of `runs/` to the LAN. Don't do that on a shared network without a firewall
or SSH tunnel (`ssh -L 9000:localhost:9000 <box>` is the intended remote
path).

## Layout

```
falsify_gui/
  app.py            FastAPI factory: routers, static mounts, reaper lifespan
  paths.py          repo paths + runs/-confined path resolution (403 on escape)
  envsetup.py       subprocess construction (env.sh wrapper, process groups)
  jobs/
    definitions.py  ← the job-type table. Adding a workflow to the GUI = one
                      entry here (form fields + argv builder + progress reader)
    manager.py      spawn / reap / kill (SIGINT → SIGKILL) / adopt-on-restart
    progress.py     filesystem+log progress readers for long-running types
    store.py        sqlite registry
  services/
    configs_enum.py configs/ family enumeration (prompts are registry-only)
    bridge.py       pi_local_bridge admin proxy (urllib, like switch.py)
    artifacts.py    runs/ browsing: campaign/recovery indices, trial detail
    npz_plot.py     trajectory NPZ → cached self-contained Plotly HTML
    datasets.py     LeRobot dataset browser (v2.1 atomic_datasets + v3.0
                    no_3pov_v3): episode lists, frame-scrubber PNGs decoded
                    straight from the parquets, state/action episode plots
  static/           index.html + app.js (vanilla, no build step) + app.css
gui/data/           runtime state: jobs.db, per-job logs, HTML caches (gitignored)
```

## Notes / caveats

- GPU-tagged job types (campaign, recovery collect, VLA episode, export,
  render-recoveries, ns-viewer) are serialized: launching while one runs
  offers to **queue** the job (sequential execution — stack N campaigns for
  an overnight sweep; the queue survives GUI restarts). The API also
  accepts `override=true` to run anyway. Jobs started from terminals are
  invisible to this guard.
- On-success **chains**: POST /api/jobs accepts `chain: [{type, args}]`;
  `"$out_dir"` in chained args is replaced with the parent's output dir.
  The UI exposes one preset: a succeeded recovery collection offers
  "Render recoveries → dataset" prefilled.
- Campaign/collection out-dirs that the scripts auto-generate are discovered
  from the `[campaign] out=` / `[collect] out=` log lines.
- Service ports are only checked against other GUI-launched services, not
  against arbitrary processes already holding the port.
- Eval campaign forms default `--no-rtc` ON (deterministic evals); prompt
  fields only offer strings from `configs/prompts/*.yaml` — never free text.
