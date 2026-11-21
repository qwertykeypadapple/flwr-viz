import * as THREE from "three";
import { Timeline, STATES, stateColor } from "./state.js";
import { FedScene, FlightIndex, RAMP_CSS } from "./scene.js";
import { lineChart } from "./charts.js";
import { explain } from "./explain.js";
import { Tour } from "./tour.js";

const $ = s => document.querySelector(s);
const hex = c => "#" + c.toString(16).padStart(6, "0");
const short = id => { const x = String(id); return x.length > 6 ? "…" + x.slice(-4) : x; };
const fmtB = b => b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : b > 1e3 ? (b / 1e3).toFixed(1) + " kB" : (b | 0) + " B";
const fmtN = v => v === null || v === undefined ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? (+v).toExponential(2) : (+v).toPrecision(4);
const PALETTE = ["#22355f", "#e76f51", "#2a9d8f", "#c99a2e", "#8e5aa8", "#3d9bd8", "#b5446e"];

// Static hosting (e.g. GitHub Pages): the page sets window.FLWR_VIZ_STATIC = {dir, manifest, default}
// and recordings are plain files; there is no collector, no WebSocket, no live mode.
const STATIC = window.FLWR_VIZ_STATIC || null;

const app = {
  runs: new Map(),       // run_id -> {tl, ix, firstSeen}
  runId: null, userPickedRun: false,
  mode: "live", source: "collector",
  playing: true, follow: true, speed: 1, vt: 0, skipGaps: true,
  selected: null, ws: null, dragging: false, lastUi: 0,
};

const scene = new FedScene($("#view"), {
  onPick: id => { if (tour && tour.active && tour.onPick(id, scene.lastPickLayer)) return; select(id); },
  onHover: () => {},
});

// Select a node (or "server", or null): its projection appears above its head
// and the camera glides over so there is room for it.
function select(id) {
  const next = id === null || id === undefined ? null : String(id);
  if (next === app.selected && next !== null) return;
  app.selected = next; scene.focus = next;
  $("#panel").scrollTop = 0;
  hideExplain();
  document.body.classList.toggle("focused", next !== null);
  scene.focusCamera(next);
  renderPanel(true);
}

// ---------------------------------------------------------------- data in
function runFor(ev) {
  const key = ev.run_id ?? "unknown";
  let r = app.runs.get(key);
  if (!r) { r = { tl: new Timeline(), ix: new FlightIndex(), firstSeen: performance.now() }; r.tl.index = r.ix; app.runs.set(key, r); refreshRunPicker(); }
  return [key, r];
}

function ingest(events) {
  let newestStart = null;
  for (const ev of events) {
    const [key, r] = runFor(ev);
    r.tl.push(ev);
    if (ev.type === "run.start") newestStart = key;
  }
  for (const r of app.runs.values()) r.ix.feed(r.tl);
  if (app.runId === null || (newestStart !== null && !app.userPickedRun && app.mode === "live")) {
    const pick = newestStart ?? [...app.runs.keys()].pop();
    if (pick !== app.runId) selectRun(pick, false);
  }
}

function selectRun(key, byUser = true) {
  app.runId = key; if (byUser) app.userPickedRun = true;
  const r = app.runs.get(key);
  scene.layoutN = -1; app.selected = null; scene.focus = null; scene.camGoal = null;
  if (r) app.vt = app.mode === "live" && app.follow ? r.tl.t1 : r.tl.t0;
  $("#run-pick").value = String(key);
  renderPanel(true); drawMarks();
}

function refreshRunPicker() {
  const sel = $("#run-pick");
  sel.innerHTML = [...app.runs.keys()].map(k => `<option value="${k}">run ${k}</option>`).join("");
  if (app.runId !== null) sel.value = String(app.runId);
  sel.parentElement.hidden = app.runs.size < 2;
}

function connect() {
  if (app.ws) try { app.ws.close(); } catch {}
  const u = new URL("ws", location.href); u.protocol = location.protocol === "https:" ? "wss:" : "ws:"; u.search = "";
  const ws = app.ws = new WebSocket(u);
  setConn("connecting");
  ws.onmessage = m => {
    const msg = JSON.parse(m.data);
    if (msg.kind === "hello") {
      app.mode = msg.mode === "replay" ? "replay" : "live";
      app.source = msg.mode === "replay" ? "collector-replay" : "collector";
      if (app.mode === "replay") { app.follow = false; app.playing = true; }
      setConn(msg.mode);
    } else if (msg.kind === "history" || msg.kind === "events") {
      ingest(msg.events);
      if (msg.kind === "history" && msg.done && app.mode === "replay") {
        const r = app.runs.get(app.runId); if (r) app.vt = r.tl.t0;
      }
    }
  };
  ws.onclose = () => { if (app.ws === ws && app.source === "collector") { setConn("offline"); setTimeout(connect, 1500); } };
}

function setConn(s) {
  const el = $("#conn");
  el.dataset.state = s;
  el.textContent = { live: "live", connecting: "connecting…", offline: "collector offline – retrying", replay: "replay", file: "recording" }[s] || s;
}

async function loadText(text, name) {
  if (app.ws) { app.source = "file"; const w = app.ws; app.ws = null; w.close(); }
  app.runs.clear(); app.runId = null; app.userPickedRun = false;
  app.mode = "replay"; app.follow = false; app.playing = true; app.source = "file";
  const evs = [];
  for (const line of text.split("\n")) { const s = line.trim(); if (!s) continue; try { evs.push(JSON.parse(s)); } catch {} }
  if (evs.every(e => typeof e.seq === "number")) evs.sort((a, b) => a.seq - b.seq);
  ingest(evs);
  const r = app.runs.get(app.runId); if (r) app.vt = r.tl.t0;
  setConn("file"); $("#src-name").textContent = name;
  $("#live-btn").hidden = false; $("#live-btn").textContent = "back to live";
}

async function openRecorded(name) {
  const res = await fetch(STATIC ? STATIC.dir + encodeURIComponent(name) : "api/runs/" + encodeURIComponent(name));
  if (!res.ok) return alert("could not load " + name);
  loadText(await res.text(), name);
}

// ---------------------------------------------------------------- timeline
function tl() { const r = app.runs.get(app.runId); return r ? r.tl : null; }

function drawMarks() {
  const t = tl(), box = $("#marks");
  if (!t || t.t1 <= t.t0) { box.innerHTML = ""; return; }
  const span = t.t1 - t.t0;
  box.innerHTML = t.roundMarks().map(m =>
    `<button class="mark" style="left:${(m.t - t.t0) / span * 100}%" data-t="${m.t}" title="jump to round ${m.round}">${m.round}</button>`).join("");
  app.marksFor = [t.events.length, t.t1];
}

function jumpRound(dir) {
  const t = tl(); if (!t) return;
  const marks = t.roundMarks(); if (!marks.length) return;
  app.follow = false;
  if (dir > 0) { const m = marks.find(m => m.t > app.vt + 1e-6); app.vt = m ? m.t : t.t1; }
  else { const prev = marks.filter(m => m.t < app.vt - 0.3); app.vt = prev.length ? prev[prev.length - 1].t : t.t0; }
}

$("#play").onclick = () => { app.playing = !app.playing; if (app.playing && tl() && app.vt >= tl().t1 && !app.follow) app.vt = tl().t0; };
$("#back").onclick = () => jumpRound(-1);
$("#fwd").onclick = () => jumpRound(+1);
$("#speed").onchange = e => { app.speed = +e.target.value; };
$("#skip").onchange = e => { app.skipGaps = e.target.checked; };
$("#live-btn").onclick = () => {
  if (app.source === "file") { app.runs.clear(); app.runId = null; app.source = "collector"; app.mode = "live"; $("#src-name").textContent = ""; connect(); }
  app.follow = true; app.playing = true;
};
$("#marks").onclick = e => { const b = e.target.closest(".mark"); if (b) { app.follow = false; app.vt = +b.dataset.t; } };
const slider = $("#scrub");
slider.oninput = () => { const t = tl(); if (!t) return; app.follow = false; app.dragging = true; app.vt = t.t0 + (t.t1 - t.t0) * slider.value / 1000; };
slider.onchange = () => { app.dragging = false; };
$("#run-pick").onchange = e => selectRun(isNaN(+e.target.value) ? e.target.value : +e.target.value);
$("#open-btn").onclick = async () => {
  const menu = $("#open-menu"); menu.hidden = !menu.hidden; if (menu.hidden) return;
  let runs = [];
  try { runs = (await (await fetch(STATIC ? STATIC.manifest : "api/runs")).json()).runs; } catch {}
  $("#open-list").innerHTML = runs.length ? runs.map(r =>
    `<li><button data-name="${r.name}">${r.title || r.name}<small>${r.title ? r.name + " · " : ""}${fmtB(r.bytes)}${r.mtime ? " · " + new Date(r.mtime * 1000).toLocaleString() : ""}</small></button></li>`).join("")
    : "<li class=muted>no recordings on this collector</li>";
};
$("#open-list").onclick = e => { const b = e.target.closest("button[data-name]"); if (b) { $("#open-menu").hidden = true; openRecorded(b.dataset.name); } };
$("#file-in").onchange = async e => { const f = e.target.files[0]; if (f) { $("#open-menu").hidden = true; loadText(await f.text(), f.name); } };
document.addEventListener("dragover", e => e.preventDefault());
document.addEventListener("drop", async e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadText(await f.text(), f.name); });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); $("#play").click(); }
  if (e.code === "ArrowRight") jumpRound(+1);
  if (e.code === "ArrowLeft") jumpRound(-1);
  if (e.code === "Escape") select(null);
});
$("#panel-close").onclick = () => select(null);

// ---------------------------------------------------------------- legend
$("#legend-states").innerHTML = STATES.map(s => `<span><i style="background:${hex(stateColor(s))}"></i>${s}</span>`).join("");
$("#ramp").style.background = `linear-gradient(90deg, #4b5568 0 8%, ${RAMP_CSS.map((c, i) => `${c} ${8 + i * 92 / (RAMP_CSS.length - 1)}%`).join(",")})`;

// ---------------------------------------------------------------- panels
function metricSeries(byRound, prefix = "") {
  const keys = new Set();
  for (const m of Object.values(byRound)) for (const k of Object.keys(m || {})) if (typeof m[k] === "number" && k !== "num-examples" && k !== "num_examples") keys.add(k);
  return [...keys].map(k => ({ name: prefix + k, points: Object.entries(byRound).map(([r, m]) => [+r, m ? m[k] : NaN]).filter(p => Number.isFinite(p[1])).sort((a, b) => a[0] - b[0]) }));
}

// Mark an element as explainable: hovering it projects a plain-language explanation.
const X = (key, ctx = {}) => ` data-x="${key}" data-c="${encodeURIComponent(JSON.stringify(ctx))}"`;

// one chart per metric: loss and accuracy on one axis hide each other
function splitGroups(groups) {
  return groups.flatMap(g => g.series.length <= 1 ? [{ ...g, kind: g.kind || g.title }] : g.series.map(s => ({ ...g, kind: g.kind || g.title, title: `${g.title}: ${s.name}`, series: [{ ...s }] })));
}

function chartsHtml(groups, phaseOf = () => null) {
  return '<div class="charts">' + groups.map((g, i) => {
    const s0 = g.series[0] || { points: [] }, pts = s0.points || [];
    const ctx = { title: g.title, kind: g.kind || g.title, name: g.series.length === 1 ? s0.name : null,
      last: pts.length ? pts[pts.length - 1][1] : null, prev: pts.length > 1 ? pts[pts.length - 2][1] : null, phase: phaseOf(g) };
    return `<div class="chart"${X("chart", ctx)}><div class="chart-h">${g.title}${g.series.length < 2 ? "" : g.series.map((s, j) =>
      `<span class="key"><i style="background:${PALETTE[j % PALETTE.length]}"></i>${s.name}</span>`).join("")}</div><canvas data-chart="${i}"></canvas></div>`;
  }).join("") + "</div>";
}

function drawCharts(groups) {
  document.querySelectorAll("#panel canvas[data-chart]").forEach(c => {
    const g = groups[+c.dataset.chart];
    lineChart(c, g.series.map((s, j) => ({ ...s, color: PALETTE[j % PALETTE.length] })), g.opts || { xLabel: "round" });
  });
}

function layerTable(layers, deltaLabel, which) {
  if (!layers || !layers.length) return `<p class="muted">No per-layer data (level "minimal", or no message yet).</p>`;
  return `<table class="layers"><tr><th${X("col.layer")}>layer</th><th${X("col.params")}>params</th><th${X("col.norm")}>‖w‖</th><th${X("col.delta", { label: deltaLabel, which })}>${deltaLabel}</th><th${X("col.rel")}>rel</th></tr>` +
    layers.map(l => {
      const rel = l.delta_norm !== undefined && l.norm ? l.delta_norm / l.norm : null;
      return `<tr${X("layer", { name: l.name, params: l.params, norm: l.norm, delta: l.delta_norm, rel, which })}><td>${l.name}</td><td>${l.params ?? "–"}</td><td>${fmtN(l.norm)}</td><td>${fmtN(l.delta_norm)}</td><td>${rel === null ? "–" : (rel * 100).toPrecision(2) + "%"}</td></tr>`;
    }).join("") + "</table>";
}

// one explainable <dt>/<dd> pair
const row = (label, value, key, ctx) => `<dt${X(key, ctx)}>${label}</dt><dd${X(key, ctx)}>${value}</dd>`;

let lastPanelKey = "";
function renderPanel(force = false) {
  const panel = $("#panel"), body = $("#panel-body");
  const t = tl();
  if (!app.selected || !t) { panel.hidden = true; lastPanelKey = ""; hideExplain(); return; }
  const s = t.state;
  panel.hidden = false;
  let html = "", groups = [], key;
  if (app.selected === "server") {
    const sv = s.server;
    key = "S" + t.applied;
    if (!force && key === lastPanelKey) return;
    // one column per phase that actually happened (train / evaluate, or an app's own message types)
    const phases = [...new Set(Object.values(sv.roundStats).flatMap(ph => Object.keys(ph)))]
      .sort((x, y) => (x === "train" ? -2 : x === "evaluate" ? -1 : 0) - (y === "train" ? -2 : y === "evaluate" ? -1 : 0));
    const short = p => p.replace(/^query\./, "");
    const rows = Object.entries(sv.roundStats).map(([r, ph]) => {
      const cell = p => ph[p] ? `${ph[p].ok}/${ph[p].sampled}` : "–";
      const failed = phases.reduce((a, p) => a + (ph[p]?.failed || 0), 0);
      return `<tr${X("replies.row", { round: r, cells: phases.map(p => `${short(p)} ${cell(p)}`).join(", "), failed })}><td>${r}</td>${phases.map(p => ph[p] ? `<td>${ph[p].ok}/${ph[p].sampled}${ph[p].failed ? ` <b class=bad>${ph[p].failed}✕</b>` : ""}</td>` : "<td>–</td>").join("")}</tr>`;
    }).join("");
    const head = `<tr><th>round</th>${phases.map(p => `<th>${short(p)}</th>`).join("")}</tr>`;
    groups = [
      { title: "aggregated train metrics", series: metricSeries(sv.metrics.train) },
      { title: "aggregated client evaluation", series: metricSeries(sv.metrics.evaluate) },
      { title: "central evaluation (server-side)", series: metricSeries(sv.metrics.central) },
    ].filter(g => g.series.length);
    groups = splitGroups(groups);
    const deltaSeries = {};
    for (const [r, ls] of Object.entries(sv.globalDelta)) {
      const tot = Math.sqrt(ls.reduce((a, l) => a + (l.delta || 0) ** 2, 0));
      const nrm = Math.sqrt(ls.reduce((a, l) => a + (l.norm || 0) ** 2, 0));
      deltaSeries[r] = { "‖Δ global‖": tot, "relative": nrm ? tot / nrm : NaN };
    }
    if (Object.keys(deltaSeries).length) groups.push({ title: "global model change per round", kind: "global model change per round", series: metricSeries(deltaSeries).slice(0, 1) });
    const phaseOf = g => g.kind === "aggregated train metrics" ? "train" : g.kind === "aggregated client evaluation" || g.kind.startsWith("central") ? "evaluate" : null;
    html = `<h2${X("server.title")}>Server <span class="badge"${X("server.status", { status: sv.status })} style="background:${sv.status === "aggregating" ? "#ffd166" : "#7cc4ff"}">${sv.status}</span></h2>
      <dl>${row("strategy", s.run.strategy ?? "–", "strategy", { name: s.run.strategy })}${row("round", `${s.run.round ?? 0} / ${s.run.numRounds ?? "?"}`, "server.round", { round: s.run.round ?? 0, total: s.run.numRounds ?? "?" })}
      ${row("nodes seen", s.order.length, "server.nodes", { n: s.order.length })}${row("messages", `${s.counts.sent} sent · ${s.counts.replies} replies · ${s.counts.failed + s.counts.timeouts} failed`, "server.messages", { sent: s.counts.sent, replies: s.counts.replies, failed: s.counts.failed + s.counts.timeouts })}
      ${row("detail level", s.run.level ?? "–", "level", { level: s.run.level })}</dl>
      ${chartsHtml(groups, phaseOf)}
      <h3${X("replies")}>Replies per round (ok / sampled)</h3><table class="rounds">${head}${rows}</table>
      <h3${X("layers.global")}>Global model layers</h3>${layerTable(sv.layers, "Δ this round", "global")}`;
  } else {
    const n = s.nodes[app.selected];
    if (!n) { panel.hidden = true; return; }
    key = n.id + "|" + t.applied;
    if (!force && key === lastPanelKey) return;
    // one group per kind of job this client did: train, evaluate, or the app's own (e.g. K-means steps)
    const byPhase = {};
    for (const [r, m] of Object.entries(n.metrics)) for (const [ph, vals] of Object.entries(m)) (byPhase[ph] ||= {})[r] = vals;
    const order = ph => ph === "train" ? 0 : ph === "evaluate" ? 1 : 2;
    groups = Object.keys(byPhase).sort((a, b) => order(a) - order(b) || a.localeCompare(b)).map(ph => ({
      title: ph === "train" ? "local train metrics" : ph === "evaluate" ? "local evaluation" : `${ph} results`,
      kind: ph === "train" ? "local train metrics" : ph === "evaluate" ? "local evaluation" : "custom job",
      phase: ph, series: metricSeries(byPhase[ph]),
    })).filter(g => g.series.length);
    groups = splitGroups(groups);
    const dur = {};
    for (const d of n.durations.train || []) if (d.round !== null) (dur[d.round] ||= {}).train_s = d.s;
    for (const d of n.durations.evaluate || []) if (d.round !== null) (dur[d.round] ||= {}).eval_s = d.s;
    if (Object.keys(dur).length) groups.push({ title: "handler duration (s)", kind: "handler duration (s)", series: metricSeries(dur) });
    if (n.progress.length) {
      const t0 = n.progress[0].t;
      groups.push({ title: "live training loss (report_progress)", kind: "live training loss (report_progress)", series: [{ name: "loss", points: n.progress.map(p => [p.t - t0, p.loss]) }], opts: { xLabel: "s", xInt: false } });
    }
    const phaseOf = g => g.phase === "train" || g.phase === "evaluate" ? g.phase : null;
    const last = n.progress[n.progress.length - 1];
    const tr = n.durations.train || [], evd = n.durations.evaluate || [];
    const avg = a => a.length ? a.reduce((x, d) => x + (d.s || 0), 0) / a.length : null;
    const rounds = Object.keys(n.metrics).map(Number).sort((a, b) => b - a);
    const lastRound = rounds[0];
    const lastM = lastRound !== undefined ? n.metrics[lastRound] : null;
    const prevOf = (ph, k) => { for (const r of rounds.slice(1)) { const v = n.metrics[r]?.[ph]?.[k]; if (typeof v === "number") return v; } return null; };
    const mrows = lastM ? Object.entries(lastM).flatMap(([ph, m]) => Object.entries(m || {}).filter(([, v]) => typeof v === "number")
      .map(([k, v]) => `<tr${X("metric", { name: k, value: v, phase: ph, prev: prevOf(ph, k) })}><td>${ph}</td><td>${k}</td><td>${fmtN(v)}</td></tr>`)).join("") : "";
    const since = n.since !== null && n.since !== undefined ? Math.max(0, t.state.run.t - n.since) : null;
    html = `<h2${X("client.title", { id: n.id })}>Client ${short(n.id)} <span class="badge"${X("state", { state: n.state, since })} style="background:${hex(stateColor(n.state))}">${n.state}</span></h2>
      <dl>${row("node id", n.id, "client.id")}
      ${row("state", `${n.state}${since !== null ? ` for ${since.toFixed(1)} s` : ""}${n.mtype ? ` · last message: ${n.mtype}` : ""}`, "client.state", { state: n.state, mtype: n.mtype })}
      ${row("rounds", `${tr.length} trained · ${evd.length} evaluated`, "client.rounds", { tr: tr.length, ev: evd.length })}
      ${row("messages", `${n.msgs} received${n.failures ? ` · <b class=bad>${n.failures} failed</b>` : ""}${n.timeouts ? ` · ${n.timeouts} timed out` : ""}`, "client.messages", { msgs: n.msgs, failures: n.failures, timeouts: n.timeouts })}
      ${row("traffic", `${fmtB(n.bytesIn)} in · ${fmtB(n.bytesOut)} out`, "client.traffic", { in: n.bytesIn, out: n.bytesOut })}
      ${row("time", `${avg(tr) !== null ? `train ${avg(tr).toFixed(2)} s avg` : "–"}${avg(evd) !== null ? ` · eval ${avg(evd).toFixed(2)} s avg` : ""}`, "client.time", { tr: avg(tr), ev: avg(evd) })}
      ${last ? row("progress", `epoch ${last.epoch ?? "–"} · batch ${(last.batch ?? 0) + 1}${last.total ? "/" + last.total : ""}${last.loss !== null && last.loss !== undefined ? " · loss " + fmtN(last.loss) : ""}`, "client.progress", { epoch: last.epoch, batch: (last.batch ?? 0) + 1, total: last.total, loss: last.loss }) : ""}
      ${n.lastError ? `<dt${X("client.error", { err: n.lastError })}>last error</dt><dd class=bad${X("client.error", { err: n.lastError })}>${n.lastError}</dd>` : ""}</dl>
      ${n.progress.length ? "" : `<p class="muted small"${X("client.blackbox")}>Local training is a black box to the mod. Call <code>report_progress()</code> in the training loop to see epochs, batches and loss here.</p>`}
      ${mrows ? `<h3${X("metrics.section", { round: lastRound })}>Metrics, round ${lastRound}</h3><table class="kv"><tr><th>phase</th><th>metric</th><th>value</th></tr>${mrows}</table>` : ""}
      ${chartsHtml(groups, phaseOf)}
      <h3${X("layers.sent")}>Update sent (last round)</h3>${layerTable(n.layers, "‖sent − received‖", "sent")}
      <h3${X("layers.recv")}>Model received</h3>${layerTable(n.recvLayers, "Δ", "recv")}`;
  }
  lastPanelKey = key;
  body.innerHTML = html;
  drawCharts(groups);
  reanchorExplain();
}

function renderHud(s, t) {
  const r = s.run;
  $("#hud-run").textContent = r.id !== null ? `run ${r.id}` : "waiting for a run…";
  $("#hud-strategy").textContent = r.strategy ?? "–";
  $("#hud-round").textContent = r.round ? `${r.round}${r.numRounds ? " / " + r.numRounds : ""}` : "–";
  $("#hud-phase").textContent = r.ended ? (r.error ? "failed" : "finished") : (r.phase ?? "–");
  const counts = {}; for (const id of s.order) counts[s.nodes[id].state] = (counts[s.nodes[id].state] || 0) + 1;
  $("#hud-nodes").textContent = s.order.length;
  $("#hud-states").innerHTML = STATES.filter(x => counts[x]).map(x => `<span><i style="background:${hex(stateColor(x))}"></i>${counts[x]} ${x}</span>`).join("");
  $("#hud-warn").textContent = r.dropped ? `${r.dropped} events were dropped by emitters` : (r.error ? r.error : "");
  $("#feed").innerHTML = s.feed.slice(-9).reverse().map(f =>
    `<li class="${f.kind}">${f.round ? `<b>R${f.round}</b>` : ""}${f.text}</li>`).join("") || `<li class="muted">Start a Flower run with viz_mod + instrument() to see events here.</li>`;
  // timeline bar
  const span = Math.max(1e-9, t.t1 - t.t0);
  if (!app.dragging) slider.value = String(Math.round((app.vt - t.t0) / span * 1000));
  $("#clock").textContent = `${fmtTime(app.vt - t.t0)} / ${fmtTime(t.t1 - t.t0)}`;
  $("#play").textContent = app.playing ? "❚❚" : "▶";
  $("#play").title = app.playing ? "pause (space)" : "play (space)";
  $("#live-btn").hidden = app.source === "collector-replay" || (app.mode === "live" && app.follow);
  if (app.source !== "file") $("#live-btn").textContent = "go live";
  if (!app.marksFor || app.marksFor[0] !== t.events.length) drawMarks();
}
const fmtTime = s => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`; };

// ---------------------------------------------------------------- labels
const labels = $("#labels");
function renderLabels(s) {
  const want = [];
  const N = s.order.length;
  if ((s.run.id !== null || N) && app.selected !== "server") want.push({ id: "server", text: `server · ${s.server.status}`, v: [0, (scene.serverTop || 1) + 1.1, 0], cls: "srv" });
  for (const id of s.order) {
    const sid = String(id);
    if (sid === app.selected) continue;              // its projection is showing instead
    if (N <= 16 || sid === scene.hover) {
      const p = scene.pos(sid); if (!p) continue;
      const n = s.nodes[id];
      want.push({ id: sid, text: sid === scene.hover || sid === app.selected ? `${short(id)} · ${n.state}` : short(id), v: [p.x, p.y + 3.25, p.z], cls: sid === app.selected ? "sel" : "" });
    }
  }
  let html = "";
  for (const w of want) {
    const q = scene.project(new THREE.Vector3(w.v[0], w.v[1], w.v[2]));
    if (!q.visible) continue;
    html += `<div class="lbl ${w.cls}" style="transform:translate(${q.x}px,${q.y}px)">${w.text}</div>`;
  }
  labels.innerHTML = html;
}

// ---------------------------------------------------------------- projection card
// The detail card hangs from the top of the light beam above the selected head.
function placeCard() {
  const panel = $("#panel");
  if (panel.hidden || !app.selected) return;
  const a = scene.anchor(app.selected);
  if (!a) return;
  const q = scene.project(a);
  const W = window.innerWidth, H = window.innerHeight;
  const w = panel.offsetWidth;
  const top = $("#top").getBoundingClientRect().bottom + 10;
  const bottom = Math.min(H - 96, Math.max(top + 200, q.y - 10));
  panel.style.maxHeight = (bottom - top) + "px";
  const h = panel.offsetHeight;
  const room = ex.el ? EX_W + 34 : 0;   // keep space beside the card for the explanation
  const left = Math.max(12, Math.min(W - w - 12 - room, q.x - w / 2));   // never off-screen; on narrow screens the explanation overlaps instead
  panel.style.left = left + "px";
  panel.style.top = (bottom - h) + "px";
  panel.style.setProperty("--tail", Math.min(w - 22, Math.max(22, q.x - left)) + "px");
  panel.classList.toggle("off", !q.visible);
}

// ---------------------------------------------------------------- explanation projection
// Hovering anything in the card projects a plain-language explanation sideways,
// on a horizontal beam from the hovered item.
const EX_W = 270;
const ex = { el: null, key: null, nth: 0 };
const exCard = $("#explain"), exBeam = $("#explain-beam"), exPoly = $("#explain-beam polygon");

function explainHtml(el) {
  let c = {};
  try { c = JSON.parse(decodeURIComponent(el.dataset.c || "%7B%7D")); } catch {}
  const r = explain(el.dataset.x, c);
  if (!r) return null;
  return `<h4>${r.title}</h4><p class="simple">${r.simple}</p>${r.here ? `<p class="here"><b>Here:</b> ${r.here}</p>` : ""}${r.tip ? `<p class="tip">${r.tip}</p>` : ""}`;
}

function showExplain(el) {
  if (ex.el === el) return;
  const html = explainHtml(el);
  if (!html) return hideExplain();
  if (ex.el) ex.el.classList.remove("xhi");
  ex.el = el; ex.key = el.dataset.x;
  ex.nth = [...$("#panel").querySelectorAll(`[data-x="${ex.key}"]`)].indexOf(el);
  el.classList.add("xhi");
  exCard.innerHTML = html; exCard.hidden = false; exBeam.style.display = "";
  exCard.classList.remove("in"); void exCard.offsetWidth; exCard.classList.add("in");
}

function hideExplain() {
  if (ex.el) ex.el.classList.remove("xhi");
  ex.el = null; ex.key = null;
  exCard.hidden = true; exBeam.style.display = "none";
}

// The card re-renders as new events arrive; find the same item again and refresh its numbers.
function reanchorExplain() {
  if (!ex.key) return;
  const el = [...$("#panel").querySelectorAll(`[data-x="${ex.key}"]`)][ex.nth];
  if (!el) return hideExplain();
  ex.el = el; el.classList.add("xhi");
  const html = explainHtml(el); if (html && html !== exCard.innerHTML) exCard.innerHTML = html;
}

$("#panel").addEventListener("mouseover", e => {
  const el = e.target.closest("[data-x]");
  if (el && $("#panel").contains(el)) showExplain(el);
});
$("#panel").addEventListener("mouseleave", hideExplain);

function placeExplain() {
  if (!ex.el || exCard.hidden) return;
  const P = $("#panel").getBoundingClientRect(), E = ex.el.getBoundingClientRect();
  const W = window.innerWidth, H = window.innerHeight;
  // the hovered item, clipped to the card's visible area
  const y0 = Math.max(E.top, P.top + 4), y1 = Math.min(E.bottom, P.bottom - 4);
  if (y1 <= y0) { exCard.style.opacity = 0; exBeam.style.opacity = 0; return; }
  exCard.style.opacity = 1; exBeam.style.opacity = 1;
  const rightRoom = W - 12 - P.right, leftRoom = P.left - 12;
  const right = rightRoom >= EX_W + 22 || rightRoom >= leftRoom;
  const w = Math.min(EX_W, Math.max(200, (right ? rightRoom : leftRoom) - 22));
  exCard.style.width = w + "px";
  const h = exCard.offsetHeight;
  const top = $("#top").getBoundingClientRect().bottom + 8;
  const cy = (y0 + y1) / 2;
  const ty = Math.min(H - 96 - h, Math.max(top, cy - h / 2));
  const x = right ? Math.min(W - 12 - w, P.right + 22) : Math.max(12, P.left - 22 - w);
  exCard.style.left = x + "px"; exCard.style.top = ty + "px"; exCard.style.transformOrigin = right ? "left center" : "right center";
  // beam: narrow at the hovered item, as tall as the card where it lands
  const sx = right ? P.right - 2 : P.left + 2, tx = right ? x + 2 : x + w - 2;
  exPoly.setAttribute("points", `${sx},${y0} ${tx},${ty + 6} ${tx},${ty + h - 6} ${sx},${y1}`);
  $("#explain-dot").setAttribute("cx", sx); $("#explain-dot").setAttribute("cy", cy);
}

// ---------------------------------------------------------------- learn mode
function setSpeed(v) {
  app.speed = v;
  const sel = $("#speed"), opts = [...sel.options].map(o => +o.value);
  sel.value = String(opts.reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a));
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => { t.hidden = true; }, 3500);
}
const tourApi = {
  tl: () => tl(),
  seek(t) { app.follow = false; app.playing = false; app.stopAt = null; app.vt = t; },
  playTo(t) {                                   // play from here to t in about 3.5 s, then pause
    app.follow = false;
    const span = t - app.vt;
    if (span <= 0) { app.vt = t; app.playing = false; return; }
    setSpeed(Math.min(16, Math.max(.5, span / 3.5)));
    app.stopAt = t; app.playing = true;
  },
  overview() { if (app.selected) select(null); scene.tourFocus = null; scene.focusCamera(null); },
  focus(id) { if (app.selected) select(null); scene.tourFocus = String(id); scene.highlightIds = new Set([String(id)]); scene.focusCamera(String(id)); },
  select(id) { select(id); },
  chests(on) { scene.showChests = on; },
  lens(on) { scene.setLens(on); },
  highlight(ids) { scene.highlightIds = new Set((ids || []).map(String)); },
  flash(ok) { const el = $("#tour"); el.classList.remove("good", "bad"); void el.offsetWidth; el.classList.add(ok ? "good" : "bad"); },
  toast,
  enter() { select(null); app.follow = false; app.playing = false; },
  leave() { scene.showChests = false; scene.setLens(false); scene.highlightIds = new Set(); scene.tourFocus = null; app.stopAt = null; setSpeed(1); },
};
const tour = new Tour(tourApi);
$("#learn-btn").onclick = () => tour.active ? tour.stop() : tour.start();
window.__tour = tour;

// ---------------------------------------------------------------- loop
let lastT = performance.now(), fpsAcc = 0, fpsN = 0;
function loop(now) {
  const dt = Math.min(0.25, (now - lastT) / 1000); lastT = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 1) { $("#fps").textContent = `${Math.round(fpsN / fpsAcc)} fps`; window.__fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
  const t = tl();
  if (t && t.events.length) {
    if (app.mode === "live" && app.follow) {
      app.vt = Math.min(app.vt + dt, t.t1);
      if (app.vt < t.t1 - 3) app.vt = t.t1 - 3; // fell behind: catch up
    } else if (app.playing && !app.dragging) {
      app.vt += dt * app.speed;
      if (app.stopAt !== null && app.stopAt !== undefined && app.vt >= app.stopAt) { app.vt = app.stopAt; app.stopAt = null; app.playing = false; }
      if (app.skipGaps) {
        const i = t.upper(app.vt);
        if (i < t.times.length && t.times[i] - app.vt > 2.5 * Math.max(1, app.speed)) app.vt = t.times[i] - 0.4 * app.speed;
      }
      if (app.vt >= t.t1 + 1.5 * app.speed) {
        if (app.loopPlayback) app.vt = t.t0;
        else { app.vt = t.t1 + 1.5 * app.speed; if (app.mode !== "live") app.playing = false; else app.follow = true; }
      }
    }
    const state = t.seek(app.vt);
    scene.update(state, t, app.vt, now / 1000, app.mode === "live" && app.follow ? 1 : app.speed);
    if (now - app.lastUi > 120) { app.lastUi = now; renderHud(state, t); renderPanel(); }
    renderLabels(state);
    placeCard();
    placeExplain();
  } else if (now - app.lastUi > 500) {
    app.lastUi = now;
    $("#feed").innerHTML = `<li class="muted">Waiting for events. Start a Flower run with <code>viz_mod</code> and <code>instrument()</code>, or open a recording.</li>`;
  }
  scene.frame();
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- boot
const params = new URLSearchParams(location.search);
window.__flwrViz = app; window.__scene = scene; // for debugging / automated checks
if (params.get("embed")) { document.body.classList.add("embed"); app.loopPlayback = true; }
if (params.get("src")) {                       // any JSONL by URL, no collector needed
  fetch(params.get("src")).then(r => r.text()).then(t => loadText(t, params.get("src").split("/").pop())).then(afterLoad);
} else if (params.get("file")) openRecorded(params.get("file")).then(afterLoad);
else if (STATIC) openRecorded(STATIC.default).then(afterLoad);
else connect();
function afterLoad() {
  if (STATIC) $("#live-btn").classList.add("gone");   // nothing live to go back to
  if (params.get("tour")) setTimeout(() => tour.start(), 600);
}
if (params.get("speed")) { app.speed = +params.get("speed"); $("#speed").value = params.get("speed"); }
requestAnimationFrame(loop);
document.addEventListener("pick-server", () => select("server"));
