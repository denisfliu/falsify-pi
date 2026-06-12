/* falsify-gui frontend: vanilla JS, no build step.
 * Tabs are hash-routed; each tab module exposes render(main). */

const $ = (sel, el = document) => el.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch (e2) { detail = res.statusText; }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return res.json();
}

const state = { configs: null, jobTypes: null, bundles: [], pollTimer: null, logSource: null };

async function loadStatic() {
  [state.configs, state.jobTypes, state.bundles] = await Promise.all([
    api("/configs"), api("/jobs/types"), api("/configs/bundles"),
  ]);
}

function bundleFor(scenarioPath) {
  const name = (scenarioPath || "").split("/").pop().replace(/\.yaml$/, "");
  return state.bundles.find(b => b.scenario === name);
}

/* ------------------------------------------------------------- jobs tab */

function optionsFor(field) {
  if (field.options) return field.options.map(o => ({ value: o, label: o }));
  const src = state.configs[field.source];
  if (!src) return [];
  if (field.source === "prompts") {
    return Object.entries(src).map(([name, task]) => ({ value: name, label: `${name} — ${task}` }));
  }
  return src.map(c => ({ value: c.path, label: c.name }));
}

function renderForm(jt, container) {
  container.replaceChildren();
  const inputs = {};
  for (const f of jt.fields) {
    const wrap = el("div", { "data-field": f.name });
    wrap.append(el("label", {}, f.label + (f.required ? " *" : "")));
    let input;
    if (f.kind === "select" || f.kind === "multiselect") {
      input = el("select", f.kind === "multiselect" ? { multiple: "" } : {});
      if (!f.required && f.kind === "select") input.append(el("option", { value: "" }, "—"));
      for (const o of optionsFor(f)) {
        const opt = el("option", { value: o.value }, o.label);
        if (f.default !== undefined && o.value === f.default) opt.selected = true;
        input.append(opt);
      }
    } else if (f.kind === "checkbox") {
      input = el("input", { type: "checkbox" });
      input.checked = !!f.default;
    } else {
      input = el("input", { type: f.kind === "number" ? "number" : "text", step: "any" });
      if (f.default !== undefined && f.default !== null) input.value = f.default;
    }
    if (f.help) wrap.append(el("span", { class: "muted" }, " " + f.help));
    wrap.append(input);
    inputs[f.name] = { field: f, input, wrap };
    container.append(wrap);
  }
  const applyShowIf = () => {
    for (const { field, wrap } of Object.values(inputs)) {
      if (!field.show_if) continue;
      const visible = Object.entries(field.show_if).every(
        ([dep, val]) => inputs[dep] && inputs[dep].input.value === val);
      wrap.style.display = visible ? "" : "none";
    }
  };
  // scenario-dependent extras: bundle status hint + scene-key multiselect
  const scenario = inputs["scenario"];
  const refreshScenario = () => {
    if (!scenario) return;
    const b = bundleFor(scenario.input.value);
    let hint = $("#bundle-hint", scenario.wrap);
    if (!hint) {
      hint = el("div", { id: "bundle-hint", class: "muted" });
      scenario.wrap.append(hint);
    }
    hint.textContent = b && b.exists
      ? `bundle: ${b.n_cards_total} cards (${Object.keys(b.cards_per_scene).join(", ")})`
      : "⚠ no bundle — run “Generate eval bundles” first";
    for (const { field, input } of Object.values(inputs)) {
      if (field.source !== "bundle_scenes") continue;
      input.replaceChildren();
      for (const [key, n] of Object.entries(b ? b.cards_per_scene : {})) {
        input.append(el("option", { value: key }, `${key} (${n})`));
      }
    }
  };
  if (scenario) {
    scenario.input.addEventListener("change", refreshScenario);
    refreshScenario();
  }
  for (const { input } of Object.values(inputs)) input.addEventListener("change", applyShowIf);
  applyShowIf();
  return inputs;
}

function formValues(inputs) {
  const args = {};
  for (const [name, { field, input, wrap }] of Object.entries(inputs)) {
    if (wrap.style.display === "none") continue;
    let v;
    if (field.kind === "checkbox") v = input.checked;
    else if (field.kind === "multiselect") v = [...input.selectedOptions].map(o => o.value);
    else v = input.value;
    if (v === "" || (Array.isArray(v) && !v.length)) continue;
    if (field.kind === "number" && v !== true) v = Number(v);
    args[name] = v;
  }
  return args;
}

function jobsTab(main) {
  const formFields = el("div");
  const typeDesc = el("p", { class: "muted", style: "font-size:0.82rem; line-height:1.45; margin:0.5rem 0 0" });
  const setType = () => {
    const jt = state.jobTypes.find(t => t.name === typeSelect.value);
    typeDesc.textContent = jt.description || "";
    currentInputs = renderForm(jt, formFields);
  };
  const typeSelect = el("select", { onchange: setType });
  for (const jt of state.jobTypes.filter(t => t.kind === "job")) {
    typeSelect.append(el("option", { value: jt.name }, jt.label + (jt.gpu ? "  [GPU]" : "")));
  }
  let currentInputs = null;
  const launchMsg = el("div", { class: "muted" });
  const launchBtn = el("button", { onclick: async () => {
    launchMsg.textContent = "";
    const type = typeSelect.value;
    const args = formValues(currentInputs);
    try {
      const job = await api("/jobs", { method: "POST", body: { type, args } });
      openJobDetail(job.id);
      refreshJobs();
    } catch (e) {
      if (e.status === 409 && confirm("A GPU job is already running or queued (" +
          e.detail.running_job_id + ").\nQueue this job to run when the GPU frees up?")) {
        const job = await api("/jobs", { method: "POST", body: { type, args, queue: true } });
        openJobDetail(job.id);
        refreshJobs();
      } else {
        launchMsg.textContent = "✗ " + e.message;
      }
    }
  }}, "Launch");

  const jobTable = el("tbody");
  const detail = el("div");

  main.replaceChildren(el("div", { class: "row" },
    el("div", { class: "panel w360" },
      el("h2", {}, "Launch"),
      el("label", {}, "Job type"), typeSelect, typeDesc, formFields, launchBtn, launchMsg),
    el("div", { class: "flex1" },
      bridgePanel(),
      el("div", { class: "panel" },
        el("h2", {}, "Jobs"),
        el("table", {},
          el("thead", {}, el("tr", {},
            el("th", {}, "id"), el("th", {}, "type"), el("th", {}, "label"),
            el("th", {}, "status"), el("th", {}, "progress"), el("th", {}, "started"))),
          jobTable)),
      detail)));

  setType();
  if (state.prefill) {
    const { type, args } = state.prefill;
    state.prefill = null;
    if (state.jobTypes.some(t => t.name === type)) {
      typeSelect.value = type;
      setType();
      for (const [k, v] of Object.entries(args || {})) {
        const inp = currentInputs[k];
        if (!inp) continue;
        if (inp.field.kind === "checkbox") inp.input.checked = !!v;
        else inp.input.value = v;
        inp.input.dispatchEvent(new Event("change"));
      }
    }
  }

  async function refreshJobs() {
    const jobs = await api("/jobs?limit=50");
    jobTable.replaceChildren(...jobs.map(j => {
      const prog = j.progress && j.progress.mode === "fraction"
        ? el("div", {}, el("div", { class: "progress" },
            el("div", { style: `width:${Math.round(100 * j.progress.done / Math.max(1, j.progress.total))}%` })),
            el("span", { class: "muted" }, ` ${j.progress.done}/${j.progress.total}`))
        : el("span", { class: "muted" }, j.status === "running" ? "…" : "");
      return el("tr", { class: "clickable", onclick: () => openJobDetail(j.id) },
        el("td", {}, el("code", {}, j.id.slice(4, 24))),
        el("td", {}, j.type),
        el("td", {}, j.label || ""),
        el("td", {}, el("span", { class: "chip " + j.status }, j.status)),
        el("td", {}, prog),
        el("td", { class: "muted" }, new Date(j.created_at * 1000).toLocaleString()));
    }));
  }

  async function openJobDetail(jobId) {
    if (state.logSource) { state.logSource.close(); state.logSource = null; }
    const j = await api("/jobs/" + jobId);
    const logBox = el("div", { id: "log-console" });
    const statusChip = el("span", { class: "chip " + j.status }, j.status);
    const killBtn = el("button", { class: "danger", onclick: async () => {
      if (confirm("Kill " + j.id + "?")) { await api(`/jobs/${jobId}/kill`, { method: "POST" }); }
    }}, "Kill");
    if (j.status !== "running" && j.status !== "queued") killBtn.style.display = "none";
    if (j.status === "queued") killBtn.textContent = "Cancel";
    // chain helper: a finished recovery collection flows into rendering
    const followUps = [];
    if (j.type === "recovery_collect" && j.status === "succeeded" && j.out_dir) {
      followUps.push(el("button", { class: "secondary", onclick: () => {
        state.prefill = { type: "render_recoveries", args: {
          recovery_run_dir: j.out_dir,
          scene: j.form_args.scene, frame: j.form_args.frame,
        }};
        route();
      }}, "Render recoveries → dataset"));
    }
    detail.replaceChildren(el("div", { class: "panel" },
      el("h2", {}, "Job ", el("code", {}, j.id), " ", statusChip, " ", killBtn,
        " ", ...followUps),
      el("table", { class: "kv" },
        el("tr", {}, el("td", {}, "args"), el("td", {}, el("code", {}, JSON.stringify(j.form_args)))),
        el("tr", {}, el("td", {}, "command"), el("td", {}, el("code", {}, j.argv.join(" ")))),
        el("tr", {}, el("td", {}, "out"), el("td", {}, j.out_dir || "—")),
        el("tr", {}, el("td", {}, "log"), el("td", {}, el("code", {}, j.log_path)))),
      logBox));
    let autoscroll = true;
    logBox.addEventListener("scroll", () => {
      autoscroll = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 10;
    });
    const src = new EventSource(`/api/jobs/${jobId}/logs`);
    state.logSource = src;
    src.onmessage = ev => {
      logBox.append(ev.data + "\n");
      if (autoscroll) logBox.scrollTop = logBox.scrollHeight;
    };
    src.addEventListener("end", ev => {
      const fin = JSON.parse(ev.data);
      statusChip.textContent = fin.status;
      statusChip.className = "chip " + fin.status;
      killBtn.style.display = "none";
      src.close();
      refreshJobs();
    });
  }

  refreshJobs();
  state.pollTimer = setInterval(refreshJobs, 2000);
}

/* ------------------------------------------------------------- runs tab */

function fmtVal(v) {
  return (v === null || v === undefined) ? "?" : v;
}

function fmtBytes(n) {
  if (n > 1e9) return (n / 1e9).toFixed(1) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  if (n > 1e3) return (n / 1e3).toFixed(1) + " KB";
  return n + " B";
}

function outcomeChip(outcome) {
  const cls = outcome === "SUCCESS" ? "succeeded"
    : outcome ? "failed" : "orphaned";
  return el("span", { class: "chip " + cls }, outcome || "?");
}

function runsTab(main) {
  const content = el("div", { class: "flex1" });
  const sidebar = el("div", { class: "w360" });
  main.replaceChildren(el("div", { class: "row" }, sidebar, content));

  async function loadSidebar() {
    const [camps, recs] = await Promise.all([
      api("/runs/campaigns"), api("/runs/recovery")]);
    const byPolicy = {};
    for (const c of camps) {
      if (!byPolicy[c.policy_id]) byPolicy[c.policy_id] = [];
      byPolicy[c.policy_id].push(c);
    }
    sidebar.replaceChildren(
      el("div", { class: "panel" },
        el("h2", {}, "Eval campaigns"),
        ...Object.entries(byPolicy).map(([pid, list]) =>
          el("details", {},
            el("summary", {}, `${pid} (${list.length})`),
            ...list.map(c => el("div", { class: "clickable", style: "padding:2px 8px",
                onclick: () => openCampaign(c.path) },
              el("code", {}, c.name), " ",
              el("span", { class: "muted" },
                `${fmtVal(c.scenario)} · ${fmtVal(c.n_succeeded)}/${fmtVal(c.n_trials_total)} ok`)))))),
      el("div", { class: "panel" },
        el("h2", {}, "Recovery collections"),
        ...(() => {
          const byPolicy = {};
          for (const r of recs) {
            const pid = r.policy_id || "?", sk = r.scene_key || "?";
            if (!byPolicy[pid]) byPolicy[pid] = {};
            if (!byPolicy[pid][sk]) byPolicy[pid][sk] = [];
            byPolicy[pid][sk].push(r);
          }
          return Object.entries(byPolicy).map(([pid, scenes]) => {
            const nRuns = Object.values(scenes).reduce((a, l) => a + l.length, 0);
            const nNpz = Object.values(scenes).flat().reduce((a, r) => a + r.n_recoveries, 0);
            return el("details", {},
              el("summary", {}, `${pid} (${nRuns} runs, ${nNpz} npz)`),
              ...Object.entries(scenes).map(([sk, list]) =>
                el("details", { style: "margin-left:1rem" },
                  el("summary", {},
                    `${sk} (${list.length} runs, ${list.reduce((a, r) => a + r.n_recoveries, 0)} npz)`),
                  ...list.map(r => el("div", { class: "clickable", style: "padding:2px 8px",
                      onclick: () => openTree(r.path) },
                    el("code", {}, r.name), " ",
                    el("span", { class: "muted" }, `${r.n_recoveries} npz`))))));
          });
        })()),
      el("div", { class: "panel" },
        el("h2", {}, "Browse runs/"),
        el("button", { class: "secondary", onclick: () => openTree("runs") }, "Open tree")));
  }

  async function openCampaign(path) {
    const d = await api("/runs/campaign?path=" + encodeURIComponent(path));
    const s = d.summary || {};
    const outcomes = Object.entries(s.by_outcome || {});
    content.replaceChildren(
      el("div", { class: "panel" },
        el("h2", {}, "Campaign ", el("code", {}, path)),
        el("table", {},
          el("thead", {}, el("tr", {}, el("th", {}, "outcome"), el("th", {}, "n"))),
          el("tbody", {}, ...outcomes.map(([k, v]) =>
            el("tr", {}, el("td", {}, outcomeChip(k)), el("td", {}, String(v)))),
            el("tr", {}, el("td", {}, "total"), el("td", {}, String(fmtVal(s.n_trials_total)))))),
        d.log ? el("p", {}, el("a", { href: "/files/" + d.log.replace(/^runs\//, ""), target: "_blank" }, "campaign.log")) : null),
      ...Object.entries(d.scenes).map(([scene, trials]) =>
        el("div", { class: "panel" },
          el("h2", {}, scene),
          el("div", {}, ...trials.map(t =>
            el("button", { class: "secondary", style: "margin:2px",
                onclick: () => openTrial(t.path) },
              t.name.replace("trial_", "#"), " ", outcomeChip(t.outcome)))))),
      ...d.viz_html.map(h => el("div", { class: "panel" },
        el("h2", {}, h.split("/").pop()),
        el("iframe", { class: "viz", src: "/files/" + h.replace(/^runs\//, "") }))));
  }

  async function openTrial(path) {
    const d = await api("/runs/trial?path=" + encodeURIComponent(path));
    const es = d.episode_summary || {};
    const kvRows = ["scene_key", "trial_index", "prompt", "posthoc_outcome",
      "transited", "n_states", "elapsed_s", "start_ned", "goal_ned"]
      .filter(k => es[k] !== undefined)
      .map(k => el("tr", {}, el("td", {}, k), el("td", {}, JSON.stringify(es[k]))));
    if (es.failure) kvRows.push(el("tr", {}, el("td", {}, "failure"),
      el("td", {}, JSON.stringify(es.failure))));
    if (es.recovery) kvRows.push(el("tr", {}, el("td", {}, "recovery"),
      el("td", {}, JSON.stringify(es.recovery))));

    const plotArea = el("div");
    content.replaceChildren(
      el("div", { class: "panel" },
        el("h2", {}, "Trial ", el("code", {}, path), " ", outcomeChip(es.posthoc_outcome)),
        el("table", { class: "kv" }, ...kvRows),
        el("div", {}, ...d.npzs.map(n =>
          el("button", { class: "secondary", style: "margin:2px", onclick: () => {
            plotArea.replaceChildren(el("iframe", { class: "viz",
              src: "/api/plot/npz?path=" + encodeURIComponent(n) }));
          }}, "plot " + n.split("/").pop()))),
        plotArea),
      d.mp4s.length ? el("div", { class: "panel" },
        el("h2", {}, "Flythrough"),
        ...d.mp4s.map(m => el("video", { controls: "", style: "max-width:100%",
          src: "/files/" + m.replace(/^runs\//, "") }))) : null,
      d.vla_io_queries.length ? el("div", { class: "panel" },
        el("h2", {}, `VLA queries (${d.vla_io_queries.length})`),
        ...d.vla_io_queries.map(q => el("details", {},
          el("summary", {}, q.name),
          el("div", {}, ...q.images.map(img =>
            el("a", { href: "/files/" + img.replace(/^runs\//, ""), target: "_blank" },
              el("img", { class: "thumb", loading: "lazy",
                src: "/files/" + img.replace(/^runs\//, "") }))))))) : null);
  }

  async function openTree(path) {
    const entries = await api("/runs/tree?path=" + encodeURIComponent(path));
    const crumbs = path.split("/");
    content.replaceChildren(el("div", { class: "panel" },
      el("h2", {}, ...crumbs.map((c, i) => el("span", {},
        i ? " / " : "",
        el("a", { href: "#", onclick: e => { e.preventDefault();
          openTree(crumbs.slice(0, i + 1).join("/")); } }, c)))),
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "name"), el("th", {}, "size"),
          el("th", {}, "modified"), el("th", {}, ""))),
        el("tbody", {}, ...entries.map(e2 => {
          const fileUrl = "/files/" + e2.path.replace(/^runs\//, "");
          const actions = [];
          if (e2.kind === "npz") actions.push(el("button", { class: "secondary",
            onclick: () => content.prepend(el("div", { class: "panel" },
              el("iframe", { class: "viz",
                src: "/api/plot/npz?path=" + encodeURIComponent(e2.path) }))) }, "plot"));
          if (["html", "png", "image", "video", "json", "log", "yaml"].includes(e2.kind))
            actions.push(el("a", { href: fileUrl, target: "_blank" }, "open"));
          const isCampaign = e2.is_dir && e2.name.startsWith("run-");
          return el("tr", { class: e2.is_dir ? "clickable" : "" },
            el("td", { onclick: e2.is_dir ? () => openTree(e2.path) : null },
              e2.is_dir ? "📁 " : "", e2.name),
            el("td", { class: "muted" }, e2.is_dir ? "" : fmtBytes(e2.size)),
            el("td", { class: "muted" }, new Date(e2.mtime * 1000).toLocaleString()),
            el("td", {}, ...actions,
              isCampaign ? el("button", { class: "secondary",
                onclick: () => openCampaign(e2.path) }, "campaign view") : "",
              e2.is_dir && e2.name.startsWith("trial_") ? el("button", { class: "secondary",
                onclick: () => openTrial(e2.path) }, "trial view") : ""));
        })))));
  }

  loadSidebar();
  // deep link: #runs:<path> opens a tree path directly
  const m = location.hash.match(/^#runs:(.+)$/);
  if (m) openTree(decodeURIComponent(m[1]));
}

/* --------------------------------------------------------- datasets tab */

function datasetsTab(main) {
  const content = el("div", { class: "flex1" });
  const sidebar = el("div", { class: "w360" });
  main.replaceChildren(el("div", { class: "row" }, sidebar, content));

  async function loadSidebar() {
    const dsets = await api("/datasets");
    const byGroup = {};
    for (const d of dsets) {
      if (!byGroup[d.group]) byGroup[d.group] = [];
      byGroup[d.group].push(d);
    }
    sidebar.replaceChildren(...Object.entries(byGroup).map(([g, list]) =>
      el("div", { class: "panel" },
        el("h2", {}, g, " ", el("span", { class: "muted" }, `(${list.length})`)),
        ...list.map(d => el("div", { class: "clickable", style: "padding:3px 8px",
            onclick: () => openDataset(d) },
          el("code", {}, d.name), el("br"),
          el("span", { class: "muted" },
            `${d.version || "?"} · ${fmtVal(d.total_episodes)} eps · ` +
            `${fmtVal(d.total_frames)} frames · ${fmtVal(d.fps)} fps`))))));
  }

  async function openDataset(d) {
    const eps = await api("/datasets/episodes?path=" + encodeURIComponent(d.path));
    content.replaceChildren(
      el("div", { class: "panel" },
        el("h2", {}, el("code", {}, d.name)),
        el("table", { class: "kv" },
          el("tr", {}, el("td", {}, "path"), el("td", {}, el("code", {}, d.path))),
          el("tr", {}, el("td", {}, "version"), el("td", {}, d.version || "?")),
          el("tr", {}, el("td", {}, "tasks"), el("td", {},
            ...(d.tasks || []).map(t => el("div", {}, "“" + t + "”")))),
          el("tr", {}, el("td", {}, "cameras"), el("td", {},
            (d.image_columns || []).join(", ")))),
        el("h2", {}, `Episodes (${eps.length})`),
        el("div", {}, ...eps.map(e2 =>
          el("button", { class: "secondary", style: "margin:2px",
              onclick: () => openEpisode(d, e2.episode_index) },
            `#${e2.episode_index}`,
            e2.length ? el("span", { class: "muted" }, ` ${e2.length}f`) : "")))));
  }

  async function openEpisode(d, index) {
    const det = await api(`/datasets/episode?path=${encodeURIComponent(d.path)}&index=${index}`);
    const cams = det.cameras.filter(c => !c.placeholder);
    const frameUrl = (cam, f) =>
      `/api/datasets/frame.png?path=${encodeURIComponent(d.path)}&index=${index}&frame=${f}&camera=${encodeURIComponent(cam.column)}`;

    const imgs = cams.map(c => el("img", {
      src: frameUrl(c, 0), title: c.column,
      style: "width:256px; height:256px; border:1px solid var(--border); border-radius:6px; margin-right:8px",
    }));
    const frameLabel = el("code", {}, `frame 0/${det.n_frames - 1}`);
    let pending = null;
    const slider = el("input", {
      type: "range", min: 0, max: det.n_frames - 1, value: 0,
      style: "width:100%",
      oninput: ev => {
        const f = Number(ev.target.value);
        frameLabel.textContent =
          `frame ${f}/${det.n_frames - 1}` +
          (det.fps ? ` · t=${(f / det.fps).toFixed(2)}s` : "");
        // throttle: only update imgs once the previous frame loaded
        if (pending !== null) { pending = f; return; }
        pending = f;
        const load = f2 => {
          let remaining = imgs.length;
          imgs.forEach((im, i) => {
            im.onload = im.onerror = () => {
              remaining -= 1;
              if (remaining === 0) {
                if (pending !== f2) load(pending); else pending = null;
              }
            };
            im.src = frameUrl(cams[i], f2);
          });
        };
        load(f);
      },
    });

    content.replaceChildren(
      el("div", { class: "panel" },
        el("h2", {}, el("code", {}, d.name), ` · episode ${index} `,
          el("button", { class: "secondary",
            onclick: () => openDataset(d) }, "← episodes")),
        det.task ? el("p", { class: "muted" }, "“" + det.task + "”") : null,
        el("p", { class: "muted" },
          `${det.n_frames} frames · ${fmtVal(det.fps)} fps · state[${det.state_dim}] · action[${det.action_dim}]` +
          (det.cameras.some(c => c.placeholder)
            ? ` · placeholders hidden: ${det.cameras.filter(c => c.placeholder).map(c => c.column).join(", ")}`
            : "")),
        el("div", { style: "display:flex" }, ...imgs),
        el("div", { style: "margin-top:8px" }, slider, frameLabel)),
      el("div", { class: "panel" },
        el("iframe", { class: "viz",
          src: `/api/datasets/plot?path=${encodeURIComponent(d.path)}&index=${index}` })));
  }

  loadSidebar();
}

/* ------------------------------------------- bridge panel (jobs page) */

async function gpuJobRunning() {
  const jobs = await api("/jobs?status=running");
  const gpuTypes = new Set(state.jobTypes.filter(t => t.gpu).map(t => t.name));
  const j = jobs.find(j2 => gpuTypes.has(j2.type));
  return j ? j.id : null;
}

function bridgePanel() {
  const box = el("div", { class: "panel" }, "loading bridge state…");

  async function refresh() {
    let d;
    try { d = await api("/bridge/policies"); }
    catch (e) { d = { reachable: false, error: e.message }; }
    if (!d.reachable) {
      box.replaceChildren(
        el("div", { class: "banner" },
          `bridge offline / unreachable: ${d.error || "?"}`,
          d.key_present === false ? " — no API key in the GUI server env (set PI_BRIDGE_API_KEYS or PI_API_KEY before starting falsify-gui)" : ""),
        el("button", { class: "secondary", onclick: refresh }, "Retry"));
      return;
    }
    const rows = (d.policies || []).map(p => {
      const tr = p.traceability || {};
      const isActive = p.is_active || p.policy_id === d.active_policy_id;
      const switchBtn = isActive ? "" : el("button", { class: "secondary", onclick: async ev => {
        const gpuJob = await gpuJobRunning();
        let msg = `Switch active checkpoint to ${p.policy_id}?`;
        if (gpuJob) msg += `\n\n⚠ GPU job ${gpuJob} is RUNNING — switching mid-rollout corrupts it.`;
        if (!confirm(msg)) return;
        ev.target.disabled = true;
        ev.target.textContent = "switching… (cold load can take minutes)";
        const res = await api("/bridge/switch", { method: "POST", body: { policy_id: p.policy_id } });
        if (!res.ok) alert("switch failed: " + res.error);
        refresh();
      }}, "Switch to");
      return el("tr", {},
        el("td", {}, el("code", {}, p.policy_id),
          isActive ? el("span", { class: "chip succeeded", style: "margin-left:6px" }, "active") : ""),
        el("td", {}, p.yaml_name || el("span", { class: "muted" }, "no local yaml")),
        el("td", { class: "muted" }, tr.variant || ""),
        el("td", { class: "muted" }, tr.notes || ""),
        el("td", {}, switchBtn));
    });
    box.replaceChildren(el("details", {},
      el("summary", {}, "Bridge ", el("code", {}, d.admin_url),
        " — active: ", el("code", {}, d.active_policy_id || "?"),
        " ", el("button", { class: "secondary",
          onclick: ev => { ev.preventDefault(); refresh(); } }, "Refresh")),
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "policy id"), el("th", {}, "yaml"),
          el("th", {}, "variant"), el("th", {}, "notes"), el("th", {}, ""))),
        el("tbody", {}, ...rows))));
  }
  refresh();
  return box;
}

/* -------------------------------------------------------------- viz tab */

function vizTab(main) {
  const serviceTypes = state.jobTypes.filter(t => t.kind === "service");
  const oneShots = state.jobTypes.filter(t =>
    ["inspect_scene", "author_mask"].includes(t.name));
  const resultArea = el("div");

  function cacheUrl(outDir) {
    if (!outDir) return null;
    if (outDir.startsWith("gui/data/cache/")) return "/gui-cache/" + outDir.slice("gui/data/cache/".length);
    if (outDir.startsWith("runs/")) return "/files/" + outDir.slice(5);
    return null;
  }

  async function renderServices(container) {
    const running = await api("/jobs?status=running");
    container.replaceChildren(...serviceTypes.map(jt => {
      const mine = running.filter(j => j.type === jt.name);
      const formFields = el("div");
      const inputs = renderForm(jt, formFields);
      const msg = el("span", { class: "muted" });
      return el("div", { class: "panel" },
        el("h2", {}, jt.label, " ",
          el("span", { class: mine.length ? "chip running" : "chip orphaned" },
            mine.length ? "running" : "stopped")),
        el("p", { class: "muted" }, jt.description),
        ...mine.map(j => {
          const port = (j.url || "").startsWith("port:") ? j.url.slice(5) : null;
          const link = port ? `http://${location.hostname}:${port}`
            : cacheUrl(j.out_dir);
          return el("div", { style: "margin:4px 0" },
            el("code", {}, j.label || j.id), " ",
            link ? el("a", { href: link, target: "_blank" }, "open") : "",
            " ", el("button", { class: "danger", onclick: async () => {
              await api(`/jobs/${j.id}/kill`, { method: "POST" });
              setTimeout(() => renderServices(container), 1500);
            }}, "Stop"));
        }),
        formFields,
        el("button", { onclick: async () => {
          msg.textContent = "";
          try {
            await api("/jobs", { method: "POST",
              body: { type: jt.name, args: formValues(inputs) } });
            msg.textContent = "started — gsplat services take ~30 s to come up";
            setTimeout(() => renderServices(container), 2000);
          } catch (e) { msg.textContent = "✗ " + e.message; }
        }}, "Start"), " ", msg);
    }));
  }

  function oneShotPanel(jt) {
    const formFields = el("div");
    const inputs = renderForm(jt, formFields);
    const msg = el("span", { class: "muted" });
    return el("div", { class: "panel" },
      el("h2", {}, jt.label),
      el("p", { class: "muted" }, jt.description),
      formFields,
      el("button", { onclick: async () => {
        msg.textContent = "generating…";
        try {
          const job = await api("/jobs", { method: "POST",
            body: { type: jt.name, args: formValues(inputs) } });
          const poll = setInterval(async () => {
            const j = await api("/jobs/" + job.id);
            if (j.status === "running") return;
            clearInterval(poll);
            if (j.status === "succeeded") {
              msg.textContent = "done";
              resultArea.replaceChildren(el("div", { class: "panel" },
                el("h2", {}, j.label),
                el("iframe", { class: "viz", src: cacheUrl(j.out_dir) })));
              resultArea.scrollIntoView({ behavior: "smooth" });
            } else {
              msg.textContent = `✗ ${j.status} — see Jobs tab for the log`;
            }
          }, 2000);
        } catch (e) { msg.textContent = "✗ " + e.message; }
      }}, "Generate"), " ", msg);
  }

  const servicesBox = el("div", { class: "flex1" });
  main.replaceChildren(
    el("div", { class: "row" },
      servicesBox,
      el("div", { class: "w360" }, ...oneShots.map(oneShotPanel))),
    resultArea);
  renderServices(servicesBox);
}

/* ------------------------------------------------------- placeholder tabs */

function placeholderTab(name) {
  return main => main.replaceChildren(
    el("div", { class: "panel" }, el("h2", {}, name), el("p", { class: "muted" }, "coming soon")));
}

/* ------------------------------------------------------------- routing */

const TABS = {
  jobs: jobsTab,
  runs: runsTab,
  datasets: datasetsTab,
  viz: vizTab,
};

function route() {
  const tab = (location.hash.slice(1) || "jobs").split(":")[0];
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  if (state.logSource) { state.logSource.close(); state.logSource = null; }
  for (const a of document.querySelectorAll("nav a")) {
    a.classList.toggle("active", a.dataset.tab === tab);
  }
  (TABS[tab] || TABS.jobs)($("#main"));
}

window.addEventListener("hashchange", route);

(async () => {
  try {
    const h = await api("/health");
    $("#health").textContent = h.falsify_py_exists ? "venv ok" : "⚠ falsify venv missing";
  } catch (e2) { $("#health").textContent = "⚠ api unreachable"; }
  await loadStatic();
  route();
})();
