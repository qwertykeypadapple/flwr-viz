// Learn mode: a question-driven, top-down tour of federated learning over any
// loaded run. At each step the timeline pauses and asks a question; the learner
// answers by clicking in the scene or picking a choice; then that part of the
// run plays and is explained with the run's real numbers.
//
// planTour(tl) is pure (no DOM) so it can be tested in Node.

const MEDIAN_LIKE = /median|krum|bulyan|trimmed/i;

const f3 = v => (v === null || v === undefined || !Number.isFinite(+v)) ? "–" : +(+v).toPrecision(3) + "";
const short = id => { const x = String(id); return x.length > 6 ? "…" + x.slice(-4) : x; };
const pct = v => `${Math.round(v * 100)}%`;

// ---------------------------------------------------------------- planning
// Find the moments in a run that each question needs.
export function planTour(tl) {
  const ev = tl.events, T = tl.times;
  const idx = (pred, from = 0) => { for (let i = from; i < ev.length; i++) if (pred(ev[i], i)) return i; return -1; };
  const all = (pred) => ev.map((e, i) => [e, i]).filter(([e, i]) => pred(e, i));
  const P = { ok: ev.length > 0 };
  if (!P.ok) return P;
  const rs = ev.find(e => e.type === "run.start");
  P.strategy = rs?.data?.strategy || "unknown";
  P.gridOnly = P.strategy === "ServerApp without a Strategy";
  P.median = MEDIAN_LIKE.test(P.strategy);
  P.t0 = T[0];

  // the "showcase" round: a training round with at least 2 successful client updates; prefer round 2
  const trainStarts = all(e => e.type === "round.start" && (e.data.phase === "train" || P.gridOnly));
  const endsIn = (r, i0) => all((e, i) => i > i0 && e.type === "client.end" && e.data.ok && e.round === r);
  let pick = null;
  for (const want of [2, 1, 3, 4, 5]) {
    const hit = trainStarts.find(([e]) => e.round === want);
    if (hit && endsIn(want, hit[1]).length >= 2) { pick = hit; break; }
  }
  if (!pick) pick = trainStarts.find(([e, i]) => endsIn(e.round, i).length >= 1) || trainStarts[0] || null;
  if (pick) {
    const [e, i] = pick;
    P.round = e.round; P.roundIdx = i; P.roundT = T[i]; P.phase = e.data.phase;
    P.sampled = (e.data.sampled || []).map(String);
    P.connected = (e.data.connected || []).map(String);
    // when each sampled client started and finished
    // replies to this round's first job only (a grid-only app may send several kinds per round), one per client
    const ends = endsIn(e.round, i).filter(([x]) => !e.data.phase || x.data.mtype === e.data.phase || (x.data.mtype || "").startsWith(e.data.phase + "."));
    const seen = new Set();
    P.ends = ends.map(([x, j]) => ({ id: String(x.src.node_id), t: T[j], dur: x.data.duration_s, metrics: x.data.metrics || {}, layers: x.data.layers || [] }))
      .sort((a, b) => a.t - b.t).filter(x => !seen.has(x.id) && seen.add(x.id));
    P.firstSent = idx((x, j) => j > i && x.type === "msg.sent");
    P.firstSentT = P.firstSent >= 0 ? T[P.firstSent] : P.roundT;
    const aggI = idx((x, j) => j > i && x.type === "agg.start" && x.round === e.round && (x.data.phase === e.data.phase));
    const aggEndI = idx((x, j) => j > i && x.type === "agg.end" && x.round === e.round && (x.data.phase === e.data.phase));
    P.aggT = aggI >= 0 ? T[aggI] : null;
    P.aggEndT = aggEndI >= 0 ? T[aggEndI] : null;
    P.aggData = aggEndI >= 0 ? ev[aggEndI].data : null;
    // the client we follow: the first to finish that sent layers, else the first to finish
    const withLayers = P.ends.filter(x => x.layers.some(l => l.delta_norm !== undefined && l.norm));
    P.focus = (withLayers[0] || P.ends[0] || null);
    if (P.focus) {
      const st = idx((x, j) => j > i && x.type === "client.start" && String(x.src.node_id) === P.focus.id);
      P.focusStartT = st >= 0 ? T[st] : P.roundT;
      const ls = P.focus.layers.filter(l => l.delta_norm !== undefined && l.norm);
      if (ls.length >= 2) {
        let best = 0; ls.forEach((l, k) => { if (l.delta_norm / l.norm > ls[best].delta_norm / ls[best].norm) best = k; });
        P.bestLayer = { name: ls[best].name, rel: ls[best].delta_norm / ls[best].norm, index: P.focus.layers.indexOf(ls[best]), count: P.focus.layers.length };
      }
    }
    // example counts for the weighting question
    const ex = P.ends.map(x => ({ id: x.id, n: num(x.metrics["num-examples"] ?? x.metrics["num_examples"]) })).filter(x => x.n !== null);
    if (ex.length >= 2) {
      ex.sort((a, b) => b.n - a.n);
      P.weights = { a: ex[0], b: ex[ex.length - 1], total: ex.reduce((s, x) => s + x.n, 0), count: ex.length };
    }
  }
  // a failure or timeout anywhere in the run
  const fi = idx(e => e.type === "msg.timeout" || (e.type === "msg.reply" && e.data.ok === false) || (e.type === "client.end" && e.data.ok === false));
  if (fi >= 0) {
    const e = ev[fi];
    P.failure = { t: T[fi], id: String(e.data.dst ?? e.data.from ?? e.src.node_id), kind: e.type === "msg.timeout" ? "timeout" : "error", round: e.round,
      error: e.data.error || e.data.error_code || "" };
    const after = idx((x, j) => j > fi && (x.type === "agg.end" || x.type === "round.end"));
    P.failure.continuedT = after >= 0 ? T[after] : null;
  }
  P.timeout = (rs?.data?.timeout) ?? null;
  return P;
}

function num(v) { return v === null || v === undefined || !Number.isFinite(+v) ? null : +v; }

// ---------------------------------------------------------------- the steps
// Each step: level, title, ask(P) -> {q, choices?|pick?, check(answer) -> bool, hint}, reveal(P, api, answerWasRight) -> text
export function buildSteps(P) {
  const S = [];
  const levels = ["The federation", "A round", "One client", "Merging", "Privacy"];
  const add = (level, s) => S.push({ level, levelName: levels[level - 1], ...s });

  // ---------------- level 1: the federation
  add(1, {
    id: "where-data",
    setup: api => { api.seek(P.roundT ? P.roundT - .01 : P.t0); api.overview(); api.chests(false); },
    q: "Where is the training data?",
    sub: "Click where you think it is: on a client, or on the diamond in the middle (the server).",
    pick: "any",
    extra: [{ label: "Everywhere", value: "everywhere" }],
    check: a => a !== "server" && a !== "everywhere",
    hint: "Think of hospitals that are not allowed to share patient records with anyone.",
    reveal: api => { api.chests(true); return "Each client keeps its own data, locked in its chest. The server in the middle has none. In federated learning the data never leaves the place it was collected."; },
  });
  add(1, {
    id: "what-travels",
    setup: api => { api.seek(P.firstSentT - .05); api.chests(true); },
    q: "If the data can't move, what has to travel instead?",
    choices: [{ label: "The data", value: "data" }, { label: "The model", value: "model" }, { label: "Both", value: "both" }],
    check: a => a === "model",
    hint: "Something has to learn from every client's data without seeing it.",
    reveal: api => { api.playTo(P.firstSentT + 1.6); return "The model travels. The server sends the current model to the clients as paper planes; each client learns from its own data and sends back only what changed in the model."; },
  });

  // ---------------- level 2: a round
  if (P.round !== undefined) {
    const k = P.sampled.length, n = P.connected.length || k;
    add(2, {
      id: "sampling",
      setup: api => { api.seek(P.roundT - .01); api.overview(); },
      q: `There are ${n} clients. In this round, does the server ask all of them?`,
      choices: [{ label: `All ${n}`, value: "all" }, { label: "Only some", value: "some" }],
      check: a => (k >= n) === (a === "all"),
      hint: "With millions of phones, could the server wait for every single one, every round?",
      reveal: api => {
        api.seek(P.roundT + .02); api.highlight(P.sampled);
        return k >= n
          ? `In this run it asked all ${n} (blue shirts). Small federations often do. With thousands of clients a server only picks a fraction each round, to save time and bandwidth.`
          : `It picked ${k} of ${n} (${pct(k / n)}), shown in blue. Picking a fraction each round keeps rounds fast; the others sit this round out.`;
      },
    });
    if (P.ends.length >= 2) {
      const first = P.ends[0];
      add(2, {
        id: "first-finish",
        setup: api => { api.seek(P.firstSentT); api.highlight(P.ends.map(e => e.id)); },
        q: "Which client will send its update back first?",
        sub: "Click the character you think will be first back.",
        pick: P.ends.map(e => e.id),
        check: a => String(a) === first.id,
        hint: "Less data, a faster machine, or simply starting earlier means being back sooner.",
        reveal: api => {
          api.highlight([first.id]); api.playTo(first.t + .3);
          const ex = num(first.metrics["num-examples"] ?? first.metrics["num_examples"]);
          const last = P.ends[P.ends.length - 1];
          const back = x => f3(x.t - P.roundT);
          return `Client ${short(first.id)} was back first, ${back(first)} s into the round (its job took ${f3(first.dur)} s${ex !== null ? ` on ${ex} examples` : ""}). The last one, ${short(last.id)}, was back after ${back(last)} s. Clients don't all start at once or run at the same speed, and the server must wait for the stragglers before it can merge.`;
        },
      });
    }
    add(2, {
      id: "silent",
      setup: api => { if (P.failure) api.seek(P.failure.t - .3); },
      q: "A client crashes or goes silent. Does the round wait for it forever?",
      choices: [{ label: "Yes, it waits", value: "wait" }, { label: "No, it carries on", value: "go" }],
      check: a => a === "go",
      hint: "One broken phone shouldn't be able to stop everyone else's progress.",
      reveal: api => {
        if (P.failure) {
          api.highlight([P.failure.id]); api.playTo((P.failure.continuedT ?? P.failure.t) + .5);
          return `Here client ${short(P.failure.id)} ${P.failure.kind === "timeout" ? "did not reply in time" : `failed (${P.failure.error.split(" (caused")[0]})`}. The server gave up on it and merged the replies it did get. The red ring marks the failure.`;
        }
        return `It carries on. This run had no failures to show, but if a client errors or misses the deadline${P.timeout ? ` (${P.timeout} s here)` : ""}, the server simply merges the replies it has.`;
      },
    });
  }

  // ---------------- level 3: one client
  if (P.focus) {
    add(3, {
      id: "client-does",
      setup: api => { api.seek(P.focusStartT - .02); api.focus(P.focus.id); },
      q: "This client just got the model. What does it do with it?",
      choices: [{ label: "Tests it, nothing more", value: "test" }, { label: "Trains it on its own data", value: "train" }, { label: "Sends it straight back", value: "back" }],
      check: a => P.phase === "train" || P.gridOnly ? a === "train" : a === "test",
      hint: "This is a training round.",
      reveal: api => {
        api.playTo(P.focus.t + .1);
        return P.gridOnly
          ? `It runs the app's own job on its local data (here: ${P.phase}), and sends back only the result.`
          : `It trains: it improves the model a little using only its own data${P.focus.dur ? `, which took ${f3(P.focus.dur)} s` : ""}, then sends back the improved model. Its books (one per layer of the model) change colour to show how much each layer moved.`;
      },
    });
    if (P.bestLayer) {
      add(3, {
        id: "which-layer",
        setup: api => { api.seek(P.focus.t + .05); api.focus(P.focus.id); },
        q: "Each book is one layer of the model. Which layer did this client change the most?",
        sub: "Click a book in the stack beside the client.",
        pick: "book:" + P.focus.id,
        check: (a, layer) => layer === P.bestLayer.index,
        hint: "Redder books changed more.",
        reveal: api => {
          api.select(P.focus.id);
          return `${P.bestLayer.name} moved the most: ${pct(P.bestLayer.rel)} of its size. Small layers (like biases) often change a lot in relative terms; colour shows relative change, width shows how many numbers the layer holds.`;
        },
      });
    }
  }

  // ---------------- level 4: merging
  if (P.weights && !P.gridOnly) {
    const { a, b, total } = P.weights;
    const equal = a.n === b.n;
    add(4, {
      id: "weighting",
      setup: api => { api.seek((P.aggT ?? P.focus?.t ?? P.roundT) - .05); api.overview(); api.highlight([a.id, b.id]); },
      q: equal ? `Both highlighted clients trained on ${a.n} examples. Whose update counts more when the server merges?`
        : `Client ${short(a.id)} trained on ${a.n} examples, client ${short(b.id)} on ${b.n}. Whose update counts more when the server merges?`,
      choices: [{ label: `${short(a.id)}`, value: "a" }, { label: `${short(b.id)}`, value: "b" }, { label: "They count the same", value: "same" }],
      check: x => (P.median || equal) ? x === "same" : x === "a",
      hint: P.median ? `This run uses ${P.strategy}. Does it look at example counts at all?`
        : equal ? "Compare the two example counts in the question." : "More data usually means a more reliable update.",
      reveal: api => {
        if (P.aggT !== null) api.playTo((P.aggEndT ?? P.aggT) + 1.4);
        if (P.median) return `With ${P.strategy} they count the same: for every number in the model it takes the middle value of the clients' numbers, and ignores how much data each one had. That makes it robust to outliers.`;
        return `With ${P.strategy} the server takes a weighted average: ${short(a.id)} counts ${pct(a.n / total)} and ${short(b.id)} counts ${pct(b.n / total)} of the result, in proportion to their examples. Watch the books fly into the diamond's pile: that is the merge.`;
      },
    });
  }
  add(4, {
    id: "outlier",
    setup: api => { api.overview(); api.highlight([]); },
    q: "One client sends garbage (a bug, or an attacker). Which merge protects the model better?",
    sub: "Try it: drag the red client's value on the line below and watch both results.",
    widget: "outlier",
    choices: [{ label: "Average (FedAvg)", value: "avg" }, { label: "Median (FedMedian)", value: "median" }],
    check: a => a === "median",
    hint: "Drag the red dot far away. Which marker follows it?",
    reveal: () => "The median barely moves however far the bad value goes; the average is dragged along with it. That is why robust strategies like FedMedian, Krum or trimmed mean exist.",
  });

  // ---------------- level 5: privacy
  add(5, {
    id: "server-sees",
    setup: api => { api.overview(); api.chests(true); api.highlight([]); },
    q: "What does the server actually see? Tick everything it receives.",
    multi: [{ label: "Raw training data", value: "data" }, { label: "Labels", value: "labels" }, { label: "Model updates", value: "updates" }, { label: "Scores and metrics", value: "metrics" }],
    check: a => a.includes("updates") && a.includes("metrics") && !a.includes("data") && !a.includes("labels"),
    hint: "Remember what travelled in the paper planes.",
    reveal: api => { api.lens(true); return "Only model updates and the scores the clients choose to send. Data and labels stay locked in the chests (greyed out now: the server never sees them). Updates can still leak a little, which is why there are extra defences such as secure aggregation and differential privacy."; },
  });
  return S;
}

// ---------------------------------------------------------------- the panel
export class Tour {
  constructor(api) {
    this.api = api; this.active = false; this.i = 0; this.steps = []; this.score = 0; this.answered = 0;
    this.el = document.getElementById("tour");
  }

  start() {
    const tl = this.api.tl();
    if (!tl || !tl.events.length) { this.api.toast("Open a recording (or wait for a run) first, then start the tour."); return; }
    this.P = planTour(tl);
    this.steps = buildSteps(this.P);
    this.active = true; this.i = 0; this.score = 0; this.answered = 0;
    document.body.classList.add("touring");
    this.api.enter();
    this.show();
  }

  stop() {
    this.active = false; this.wantPick = null;
    document.body.classList.remove("touring");
    this.el.hidden = true;
    this.api.leave();
  }

  show() {
    const s = this.steps[this.i];
    if (!s) return this.finish();
    this.state = "ask"; this.firstTry = true; this.tries = 0;
    this.api.lens(false);
    s.setup?.(this.api);
    this.wantPick = s.pick ?? null;
    const dots = [1, 2, 3, 4, 5].map(l => `<i class="${l < s.level ? "done" : l === s.level ? "now" : ""}" title="${["The federation", "A round", "One client", "Merging", "Privacy"][l - 1]}"></i>`).join("");
    const choices = (s.choices || []).concat(s.extra || []);
    this.el.innerHTML = `
      <div class="t-head"><span class="t-level">Level ${s.level} · ${s.levelName}</span><span class="t-dots">${dots}</span>
        <button class="t-x" data-act="exit" title="leave the tour">×</button></div>
      <p class="t-q">${s.q}</p>
      ${s.sub ? `<p class="t-sub">${s.sub}</p>` : ""}
      ${s.pick ? `<p class="t-pick">👆 click in the scene</p>` : ""}
      ${s.widget === "outlier" ? outlierWidget() : ""}
      ${choices.length ? `<div class="t-choices">${choices.map(c => `<button data-v="${c.value}">${c.label}</button>`).join("")}</div>` : ""}
      ${s.multi ? `<div class="t-multi">${s.multi.map(c => `<label><input type="checkbox" value="${c.value}"> ${c.label}</label>`).join("")}</div><div class="t-choices"><button data-act="submit">Check</button></div>` : ""}
      <p class="t-feedback" hidden></p>
      <div class="t-foot"><button data-act="show" class="t-link">show me</button><span class="t-step">${this.i + 1} / ${this.steps.length}</span><button data-act="next" hidden>next →</button></div>`;
    this.el.hidden = false;
    this.el.classList.remove("in"); void this.el.offsetWidth; this.el.classList.add("in");
    this.el.onclick = e => this.click(e);
    if (s.widget === "outlier") wireOutlier(this.el);
  }

  click(e) {
    const b = e.target.closest("button"); if (!b) return;
    const s = this.steps[this.i];
    if (b.dataset.act === "exit") return this.stop();
    if (b.dataset.act === "next") { this.i++; return this.show(); }
    if (b.dataset.act === "show") return this.answer(null, null, true);
    if (this.state !== "ask") return;
    if (b.dataset.act === "submit") return this.answer([...this.el.querySelectorAll(".t-multi input:checked")].map(x => x.value));
    if (b.dataset.v !== undefined) return this.answer(b.dataset.v);
  }

  // called by the app when the learner clicks in the scene during a pick question
  onPick(id, layer) {
    if (!this.active || this.state !== "ask" || !this.wantPick) return false;
    const w = this.wantPick;
    if (id === null || id === undefined) { this.feedback(w === "any" ? "Click a character, or the diamond in the middle." : "Click on a character (or a book) in the scene.", "hint"); return true; }
    if (typeof w === "string" && w.startsWith("book:")) {
      if (layer === null || layer === undefined || String(id) !== w.slice(5)) { this.feedback("Click one of the books beside the highlighted client.", "hint"); return true; }
      this.answer(id, layer); return true;
    }
    if (Array.isArray(w) && id !== "server" && !w.includes(String(id))) { this.feedback("Pick one of the highlighted clients.", "hint"); return true; }
    this.answer(id === null ? null : String(id)); return true;
  }

  answer(a, layer, giveUp = false) {
    const s = this.steps[this.i];
    const right = !giveUp && s.check(a, layer);
    this.tries++;
    if (!right && !giveUp && this.tries < 2) {
      this.firstTry = false;
      this.feedback(`Not quite. ${s.hint} Try again, or press "show me".`, "hint");
      this.api.flash(false);
      return;
    }
    if (this.firstTry && right) this.score++;
    this.answered++;
    this.state = "reveal"; this.wantPick = null;
    const text = s.reveal(this.api, right);
    this.feedback(`${right ? "<b>Yes!</b> " : giveUp ? "" : "<b>Here's what happened.</b> "}${text}`, right ? "right" : "info");
    if (right) this.api.flash(true);
    this.el.querySelectorAll(".t-choices button, .t-multi input").forEach(x => x.disabled = true);
    this.el.querySelector(".t-pick")?.remove();
    const next = this.el.querySelector('[data-act="next"]'); next.hidden = false;
    this.el.querySelector('[data-act="show"]').hidden = true;
    if (this.i === this.steps.length - 1) next.textContent = "finish →";
  }

  feedback(html, kind) {
    const f = this.el.querySelector(".t-feedback");
    f.innerHTML = html; f.hidden = false; f.className = "t-feedback " + kind;
  }

  finish() {
    this.state = "done"; this.wantPick = null;
    this.api.lens(false); this.api.highlight([]); this.api.overview();
    this.el.innerHTML = `
      <div class="t-head"><span class="t-level">You made it through all five levels</span><button class="t-x" data-act="exit">×</button></div>
      <p class="t-q">${this.score} of ${this.steps.length} right on the first try.</p>
      <ol class="t-sum">
        <li><b>The federation:</b> data stays with each client; only the model travels.</li>
        <li><b>A round:</b> the server picks clients, sends the model, waits (but not forever), and collects replies.</li>
        <li><b>One client:</b> trains on its own data and sends back what changed.</li>
        <li><b>Merging:</b> the server combines updates: weighted by data (FedAvg), or robustly (FedMedian).</li>
        <li><b>Privacy:</b> the server sees updates and scores, never the data.</li>
      </ol>
      <div class="t-choices"><button data-act="again">Take the tour again</button><button data-act="exit">Explore on my own</button></div>`;
    this.el.onclick = e => { const b = e.target.closest("button"); if (!b) return; if (b.dataset.act === "again") this.start(); else this.stop(); };
  }
}

// ---------------------------------------------------------------- mean vs median mini-demo
function outlierWidget() {
  return `<div class="t-outlier"><svg viewBox="0 0 400 86" width="100%" height="86" aria-label="client values on a line">
    <line x1="10" y1="50" x2="390" y2="50" stroke="#1f1d22" stroke-width="2"/>
    <g class="pts"></g>
    <g class="mk-mean"><line y1="18" y2="62" stroke="#e76f51" stroke-width="3"/><text y="14" text-anchor="middle" fill="#e76f51">average</text></g>
    <g class="mk-med"><line y1="38" y2="82" stroke="#22355f" stroke-width="3"/><text y="80" dy="0" text-anchor="middle" fill="#22355f" class="lbl">median</text></g>
  </svg><p class="t-sub">Five honest clients sent similar updates (blue). Drag the red one.</p></div>`;
}

function wireOutlier(root) {
  const svg = root.querySelector(".t-outlier svg"); if (!svg) return;
  const honest = [150, 165, 172, 180, 192];
  let bad = 210;
  const g = svg.querySelector(".pts"), mean = svg.querySelector(".mk-mean"), med = svg.querySelector(".mk-med");
  const draw = () => {
    g.innerHTML = honest.map(x => `<circle cx="${x}" cy="50" r="8" fill="#7fb2e5" stroke="#1f1d22" stroke-width="2"/>`).join("") +
      `<circle class="bad" cx="${bad}" cy="50" r="10" fill="#e5534b" stroke="#1f1d22" stroke-width="2" style="cursor:grab"/>`;
    const vals = [...honest, bad].sort((a, b) => a - b);
    const m = vals.reduce((a, b) => a + b, 0) / vals.length, md = (vals[2] + vals[3]) / 2;
    mean.setAttribute("transform", `translate(${m},0)`); med.setAttribute("transform", `translate(${md},0)`);
    mean.querySelector("line").setAttribute("x1", 0); mean.querySelector("line").setAttribute("x2", 0);
    med.querySelector("line").setAttribute("x1", 0); med.querySelector("line").setAttribute("x2", 0);
  };
  draw();
  let drag = false;
  const toX = e => { const r = svg.getBoundingClientRect(); return Math.max(12, Math.min(388, (e.clientX - r.left) / r.width * 400)); };
  svg.addEventListener("pointerdown", e => { if (e.target.classList.contains("bad")) { drag = true; svg.setPointerCapture?.(e.pointerId); } });
  svg.addEventListener("pointermove", e => { if (drag) { bad = toX(e); draw(); } });
  svg.addEventListener("pointerup", () => { drag = false; });
}
