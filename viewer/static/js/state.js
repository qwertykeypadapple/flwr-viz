// Pure state reducer over flwr-viz v1 events. No DOM, no three.js:
// the same code drives live view, scrubbing and replay, and is tested in Node.

export const STATES = ["offline", "idle", "selected", "training", "evaluating", "working", "sending", "done", "failed"];

const MAX_PROGRESS = 400;   // progress points kept per node
const MAX_FEED = 300;       // narrated feed lines kept

export function emptyState() {
  return {
    run: { id: null, strategy: null, numRounds: null, round: 0, phase: null, level: null,
           started: null, ended: null, error: null, dropped: 0, t: null },
    server: { status: "idle", layers: [], globalDelta: {}, metrics: { train: {}, evaluate: {}, central: {} },
              roundStats: {}, aggT: null },
    nodes: {},          // id -> node
    order: [],          // node ids in first-seen order (stable layout)
    msgs: {},           // msg_id -> {dst, phase, round, t}
    feed: [],
    counts: { events: 0, sent: 0, replies: 0, failed: 0, timeouts: 0 },
  };
}

function node(s, id) {
  if (id === null || id === undefined) return null;
  let n = s.nodes[id];
  if (!n) {
    n = s.nodes[id] = {
      id, state: "idle", mtype: null, since: null, msgId: null,
      layers: [], recvLayers: [], metrics: {}, durations: { train: [], evaluate: [] },
      progress: [], progressMsg: null, msgs: 0, failures: 0, timeouts: 0,
      bytesIn: 0, bytesOut: 0, lastError: null, connected: true,
    };
    s.order.push(id);
  }
  return n;
}

function setState(n, st, t) {
  if (n.state !== st) { n.state = st; n.since = t; }
}

function feed(s, t, text, kind = "info", round = null) {
  s.feed.push({ t, text, kind, round });
  if (s.feed.length > MAX_FEED) s.feed.splice(0, s.feed.length - MAX_FEED);
}

function short(id) {
  const x = String(id);
  return x.length > 5 ? "…" + x.slice(-4) : x;
}

// "train", "evaluate", or the app's own job name ("query.perform_local_clustering" -> "perform_local_clustering")
export function phaseOf(mtype) {
  if (!mtype) return null;
  if (mtype === "train" || mtype.startsWith("train.")) return "train";
  if (mtype === "evaluate" || mtype.startsWith("evaluate.")) return "evaluate";
  return mtype.replace(/^query\./, "");
}

function fmtMetrics(m) {
  const keys = Object.keys(m || {}).filter(k => typeof m[k] === "number" && k !== "num-examples");
  return keys.slice(0, 3).map(k => `${k}=${(+m[k]).toPrecision(3)}`).join(", ");
}

export function roundOfEvent(s, ev) {
  if (ev.round !== null && ev.round !== undefined) return ev.round;
  const m = ev.msg_id && s.msgs[ev.msg_id];
  return m ? m.round : null;
}

// Apply one event (mutates s). `t` is the event's monotonic timeline time.
export function apply(s, ev, t = ev.ts) {
  s.counts.events++;
  s.run.t = t;
  const d = ev.data || {};
  const rnd = roundOfEvent(s, ev);
  switch (ev.type) {
    case "run.start": {
      s.run.id = ev.run_id; s.run.strategy = d.strategy; s.run.numRounds = d.num_rounds;
      s.run.level = d.level; s.run.started = t; s.run.ended = null; s.run.error = null;
      s.server.layers = d.layers || [];
      for (const id of d.connected || []) node(s, id);
      feed(s, t, `Run started: ${d.strategy}, ${d.num_rounds ? d.num_rounds + " rounds" : "rounds decided by the app as it goes"}, ${(d.connected || []).length} nodes connected`, "run");
      break;
    }
    case "round.start": {
      s.run.round = rnd; s.run.phase = d.phase;
      const connected = new Set((d.connected || []).map(String));
      for (const id of d.connected || []) node(s, id);
      for (const id of s.order) {
        const n = s.nodes[id];
        n.connected = connected.size === 0 || connected.has(String(id));
        if (!n.connected) setState(n, "offline", t);
        else setState(n, "idle", t);
      }
      for (const id of d.sampled || []) { const n = node(s, id); if (n) setState(n, "selected", t); }
      s.server.status = "waiting";
      const rs = s.server.roundStats[rnd] ||= {};
      rs[d.phase] = { sampled: (d.sampled || []).length, ok: 0, failed: 0 };
      feed(s, t, `Round ${rnd} · ${d.phase}: server samples ${(d.sampled || []).length} of ${(d.connected || []).length} nodes`, "round", rnd);
      break;
    }
    case "msg.sent": {
      s.counts.sent++;
      if (ev.msg_id) s.msgs[ev.msg_id] = { dst: d.dst, phase: d.phase, round: rnd, t };
      const n = node(s, d.dst);
      if (n) { n.msgs++; if (n.state === "idle" || n.state === "done" || n.state === "failed") setState(n, "selected", t); }
      break;
    }
    case "client.recv": {
      const n = node(s, ev.src.node_id);
      if (!n) break;
      n.bytesIn += d.bytes || 0;
      if (d.layers && d.layers.length) n.recvLayers = d.layers;
      n.msgId = ev.msg_id; n.mtype = d.mtype;
      break;
    }
    case "client.start": {
      const n = node(s, ev.src.node_id);
      if (!n) break;
      n.mtype = d.mtype; n.msgId = ev.msg_id;
      // train -> training, evaluate -> evaluating, anything custom (query.*, app-defined) -> working
      const ph = phaseOf(d.mtype);
      setState(n, ph === "evaluate" ? "evaluating" : ph === "train" ? "training" : "working", t);
      if (phaseOf(d.mtype) === "train") { n.progress = []; n.progressMsg = ev.msg_id; }
      break;
    }
    case "client.progress": {
      const n = node(s, ev.src.node_id);
      if (!n) break;
      if (n.progressMsg !== ev.msg_id) { n.progress = []; n.progressMsg = ev.msg_id; }
      n.progress.push({ t, epoch: d.epoch, batch: d.batch, total: d.total_batches, loss: d.loss });
      if (n.progress.length > MAX_PROGRESS) n.progress.splice(0, n.progress.length - MAX_PROGRESS);
      if (n.state === "selected") setState(n, "training", t);
      break;
    }
    case "client.end": {
      const n = node(s, ev.src.node_id);
      if (!n) break;
      const ph = phaseOf(d.mtype) || "other";
      if (d.ok) {
        setState(n, "sending", t);
        n.bytesOut += d.bytes || 0;
        if (d.layers && d.layers.length) n.layers = d.layers;
        if (rnd !== null) {
          const r = n.metrics[rnd] ||= {};
          r[ph] = d.metrics || {};
        }
        (n.durations[ph] ||= []).push({ round: rnd, s: d.duration_s });
      } else {
        setState(n, "failed", t); n.failures++; n.lastError = d.error || "error";
        feed(s, t, `Node ${short(n.id)} failed during ${ph}: ${d.error || "error"}`, "fail", rnd);
      }
      break;
    }
    case "msg.reply": {
      s.counts.replies++;
      const n = node(s, d.from);
      const rs = (s.server.roundStats[rnd] || {})[d.phase];
      if (d.ok) {
        if (n && n.state !== "failed") setState(n, "done", t);
        if (rs) rs.ok++;
      } else {
        s.counts.failed++;
        if (rs) rs.failed++;
        if (n) {
          if (n.state !== "failed") { n.failures++; }
          setState(n, "failed", t); n.lastError = d.error_code + (d.error_reason ? ": " + d.error_reason : "");
        }
        feed(s, t, `Reply from node ${short(d.from)} is an error: ${d.error_code}`, "fail", rnd);
      }
      break;
    }
    case "msg.timeout": {
      s.counts.timeouts++;
      const n = node(s, d.dst);
      const rs = (s.server.roundStats[rnd] || {})[d.phase];
      if (rs) rs.failed++;
      if (n) { setState(n, "failed", t); n.timeouts++; n.failures++; n.lastError = `timed out after ${(+d.waited_s).toFixed(1)}s`; }
      feed(s, t, `Node ${short(d.dst)} did not reply in time (${(+d.waited_s).toFixed(1)} s)`, "fail", rnd);
      break;
    }
    case "agg.start": {
      s.server.status = "aggregating"; s.server.aggT = t;
      feed(s, t, `Round ${rnd} · ${d.phase}: aggregating ${d.n_ok} replies` + (d.n_failed ? ` (${d.n_failed} failed)` : ""), "agg", rnd);
      break;
    }
    case "agg.end": {
      s.server.status = "idle";
      if (d.phase === "train" && d.layers && d.layers.length) {
        s.server.layers = d.layers;
        s.server.globalDelta[rnd] = d.layers.map(l => ({ name: l.name, delta: l.delta_norm, norm: l.norm }));
      }
      if (rnd !== null) s.server.metrics[d.phase === "train" ? "train" : "evaluate"][rnd] = d.metrics || {};
      for (const id of s.order) { const n = s.nodes[id]; if (n.state === "done" || n.state === "sending") setState(n, "idle", t); }
      const fm = fmtMetrics(d.metrics);
      feed(s, t, `Round ${rnd} · ${d.phase}: ${d.phase === "train" ? "new global model" : "aggregated evaluation"}${fm ? " — " + fm : ""}`, "agg", rnd);
      break;
    }
    case "eval.central": {
      if (rnd !== null) s.server.metrics.central[rnd] = d.metrics || {};
      const fm = fmtMetrics(d.metrics);
      feed(s, t, `Central evaluation${rnd ? " after round " + rnd : " of the initial model"}${fm ? ": " + fm : ""}`, "eval", rnd);
      break;
    }
    case "round.end": {
      feed(s, t, `Round ${rnd} finished in ${(+d.duration_s).toFixed(1)} s`, "round", rnd);
      break;
    }
    case "run.end": {
      s.run.ended = t; s.server.status = "idle";
      for (const id of s.order) { const n = s.nodes[id]; if (n.state !== "offline" && n.state !== "failed") setState(n, "idle", t); }
      feed(s, t, `Run finished: ${d.rounds_completed} rounds in ${(+d.duration_s).toFixed(1)} s`, "run");
      break;
    }
    case "run.error": {
      s.run.ended = t; s.run.error = d.error; s.server.status = "idle";
      feed(s, t, `Run failed: ${d.error}`, "fail");
      break;
    }
    case "sys.dropped": {
      s.run.dropped += d.count || 0;
      break;
    }
    default: break;
  }
  return s;
}

// -- Timeline: monotonic times, snapshots for fast seeking ----------------

export class Timeline {
  constructor(snapshotEvery = 400) {
    this.events = [];   // events of the selected run, in seq order
    this.times = [];    // monotonic timeline time per event
    this.snaps = [];    // {i, state} : state after applying events[0..i)
    this.every = snapshotEvery;
    this.state = emptyState();
    this.applied = 0;   // events applied to this.state
  }

  push(ev) {
    const prev = this.times.length ? this.times[this.times.length - 1] : -Infinity;
    // Clock skew between machines can make ts go backwards; the timeline never does.
    const t = Math.max(prev, typeof ev.ts === "number" ? ev.ts : prev);
    this.events.push(ev); this.times.push(t);
  }

  get t0() { return this.times.length ? this.times[0] : 0; }
  get t1() { return this.times.length ? this.times[this.times.length - 1] : 0; }

  // index of first event with time > t
  upper(t) {
    let lo = 0, hi = this.times.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.times[m] <= t) lo = m + 1; else hi = m; }
    return lo;
  }

  lower(t) { // first index with time >= t
    let lo = 0, hi = this.times.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.times[m] < t) lo = m + 1; else hi = m; }
    return lo;
  }

  // Bring this.state to "all events with time <= t applied".
  seek(t) {
    const target = this.upper(t);
    if (target < this.applied) {
      // go back: restore the latest snapshot at or before target
      let snap = null;
      for (let k = this.snaps.length - 1; k >= 0; k--) if (this.snaps[k].i <= target) { snap = this.snaps[k]; break; }
      this.state = snap ? structuredClone(snap.state) : emptyState();
      this.applied = snap ? snap.i : 0;
    }
    while (this.applied < target) {
      apply(this.state, this.events[this.applied], this.times[this.applied]);
      this.applied++;
      if (this.applied % this.every === 0 && !this.snaps.some(s => s.i === this.applied)) {
        this.snaps.push({ i: this.applied, state: structuredClone(this.state) });
      }
    }
    return this.state;
  }

  roundMarks() {
    const out = [];
    const seen = new Set();   // the first step of each round, whatever its phase (train, or an app's own job)
    for (let i = 0; i < this.events.length; i++) {
      const e = this.events[i];
      if (e.type === "round.start" && e.round !== null && !seen.has(e.round)) { seen.add(e.round); out.push({ round: e.round, t: this.times[i] }); }
    }
    return out;
  }
}

// Shirt colours of the characters (light theme). Idle is a plain white shirt.
export function stateColor(st) {
  return {
    offline: 0xbdb6a8, idle: 0xf7f5f0, selected: 0x7fb2e5, training: 0x5cc39a,
    evaluating: 0xa98ad6, working: 0x63c5da, sending: 0xf4a259, done: 0x9ad1b5, failed: 0xe5534b,
  }[st] ?? 0xf7f5f0;
}
