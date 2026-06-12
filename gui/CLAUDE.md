# `gui/` — falsify management GUI

FastAPI + vanilla-JS web app (own venv, modeled on `pi_local_bridge/`) that
manages the repo's workflows. Single source of truth for its contracts —
keep current like every CLAUDE.md in this repo. User docs live in README.md;
this file is for whoever (human or Claude) extends the GUI.

## Iron rules

1. **The backend never imports falsify.** All falsify work runs as
   subprocesses of the repo-root venv's python under `tools/env.sh`
   (`envsetup.build_argv`). The GUI venv has numpy/plotly/pyarrow — never
   torch. If a feature seems to need falsify imports, it actually needs a
   new CLI in falsify or a subprocess.
2. **Install GUI deps with `uv pip install -p .venv/bin/python …` from
   `gui/`.** The user's shell exports `VIRTUAL_ENV` pointing at the shared
   SousVide venv; a bare `uv pip install` silently pollutes it (this
   happened — see git history around 2026-06-12 — websockets got bumped
   past falsify's `<12` pin).
3. **Prompt form fields are registry-only** (`configs/prompts/*.yaml`,
   shape `prompts.<name>.task`). Never add a free-text prompt field; the
   strings must exist verbatim in the policy's training tasks. Builders
   resolve names via `configs_enum.prompt_task()` (raises on unknown).
4. **Eval forms default `--no-rtc` ON.** RTC is nondeterministic across
   runs; evals must be reproducible.
5. **Client-supplied paths are repo-relative and confined**:
   `paths.resolve_runs_path` (runs/) and `datasets._resolve` (data/) raise
   on escape → routers map to 403. Any new browsing endpoint must use the
   same pattern.

## Adding a workflow to the GUI

One entry in `falsify_gui/jobs/definitions.py` — nothing else:

- `fields`: rendered by the frontend generically (`select | multiselect |
  text | number | checkbox`; `source` = key in `/api/configs`;
  `show_if={"field": value}` for conditional fields; `source="bundle_scenes"`
  is special-cased to the chosen scenario's bundle scene keys).
- `build(args) -> Built(script_args, out_dir, url, label)`: `script_args`
  are handed to the falsify venv python (`-m module` or a `scripts/…` path,
  cwd = repo root). Set `out_dir=None` when the script auto-names its
  output dir.
- `progress(job) -> {mode, done, total, detail?, out_dir?}` (jobs/progress.py):
  cheap filesystem polls. Returning `out_dir` persists a discovered
  auto-named dir — campaign/collect print `[campaign] out=` / `[collect]
  out=` lines for this.
- `finalize(job)`: status inference when the exit code was lost (GUI
  restart). Directory outputs prove nothing (they exist from t=0); check a
  completion artifact (`campaign_summary.json`).
- `done_marker`: the script's final printed line. Campaign/collect scripts
  historically never exited (non-daemon gateway WS threads); the
  orchestrator now closes policies in `run_episode`'s `finally`
  (src/falsify/orchestrator/orchestrator.py), but the marker remains as a
  belt-and-suspenders reaper: marker seen + process alive 60 s later →
  SIGINT group, record `succeeded`.

## Job lifecycle invariants (jobs/manager.py)

- stdout+stderr go to `gui/data/jobs/<id>/job.log` **files, not pipes**;
  SSE tailing is file polling; jobs survive GUI restarts.
- Every job runs in its own **process group** (`start_new_session`); kill =
  SIGINT group → 10 s → SIGKILL group. Lingering-reap and kill-button paths
  must stay distinguishable (`_lingering` vs `_kill_requested` — order of
  the checks in `_finish` matters).
- Restart adoption: `status=running` rows are re-adopted iff `/proc/<pid>`
  starttime matches the stored value (pid-reuse guard). `status=queued`
  rows survive restarts untouched.
- **GPU queue**: GPU-tagged types are serialized. Launch while busy → 409;
  `queue=true` → `status=queued`, the reaper starts the oldest queued GPU
  job when the GPU frees. Sweeps = queue N campaigns.
- **Chains**: `Job.chain = [{"type", "args"}]`; on success the first entry
  launches (queued) with `"$out_dir"` in arg values replaced by the
  parent's out_dir, remaining entries passed down. UI exposes one preset:
  succeeded `recovery_collect` → prefilled `render_recoveries` form.

## Gotchas discovered the hard way

- `rollout_states.npz` `failure_type` (and recovery `prompt`/`source`) are
  pickled object arrays — `np.load(allow_pickle=False)` raises on access;
  npz_plot reads scalars defensively.
- `dagger-1_*` datasets have an **empty `features` dict** in info.json —
  datasets.py sniffs image columns from the parquet schema
  (`struct{bytes,path}`) as fallback.
- v2.1 vs v3.0 LeRobot layouts differ in `data_path` placeholder names
  (`episode_chunk` vs `chunk_index`) — `_episode_parquet` formats with both.
- esprima (used ad hoc for app.js syntax checks) only parses ≤ES2017-ish:
  no `??`/`??=`/optional catch binding in app.js, by convention.
- `pkill -f`/`pgrep -f` from a test shell matches the test shell itself —
  use `[f]oo` bracket patterns.
- The frontend is one `app.js`, no build step; tabs are functions in `TABS`,
  hash-routed (`#runs:<path>` deep-links a tree path).

## Secrets

`tools/secrets.env` (gitignored) holds `PI_API_KEY` / `PI_BRIDGE_API_KEYS`;
`__main__.load_secrets_env()` reads it at startup (explicit env wins), and
`tools/pi_inference_env.sh` sources it for terminal workflows. Spawned jobs
inherit the server env. `/api/health` reports key presence.
