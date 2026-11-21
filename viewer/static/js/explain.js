// Plain-language explanations for everything shown in a projection card.
// explain(key, ctx) -> { title, simple, here?, tip? }  (strings may contain light HTML)
// `ctx` carries the actual values of the hovered item, so the "here" line can
// say what this particular number means right now.

const num = v => (v === null || v === undefined || v === "" || !Number.isFinite(+v)) ? null : +v;
const f = (v, d = 3) => { v = num(v); if (v === null) return "–"; const a = Math.abs(v); return a >= 1e4 || (a < 1e-3 && a !== 0) ? v.toExponential(2) : +v.toPrecision(d) + ""; };
const pct = v => num(v) === null ? "–" : (+v * 100 < 0.1 ? (+v * 100).toFixed(3) : +(+v * 100).toPrecision(2)) + "%";
const bytes = b => b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : b > 1e3 ? (b / 1e3).toFixed(1) + " kB" : (b | 0) + " bytes";

function sizeWord(rel) {
  rel = num(rel); if (rel === null) return null;
  if (rel < 0.001) return "tiny: this layer barely moved";
  if (rel < 0.01) return "small: a gentle nudge";
  if (rel < 0.1) return "moderate: clearly learning";
  if (rel < 0.5) return "large: a big step, common in early rounds or with a high learning rate";
  return "very large: the layer was almost rewritten. Normal in round 1 from a random start; worrying late in training";
}

function trend(name, v, prev) {
  v = num(v); prev = num(prev);
  if (v === null || prev === null) return "";
  const lower = /loss|err|time|_s$|duration/i.test(name), higher = /acc|auc|f1|precision|recall|score/i.test(name);
  const d = v - prev;
  if (Math.abs(d) < 1e-12) return ` Same as the round before (${f(prev)}).`;
  const dir = d < 0 ? "down" : "up";
  const good = (lower && d < 0) || (higher && d > 0), bad = (lower && d > 0) || (higher && d < 0);
  return ` ${dir === "down" ? "Down" : "Up"} from ${f(prev)} the round before${good ? ": better." : bad ? ": worse this round (a single noisy round is normal)." : "."}`;
}

function layerKind(name) {
  const n = String(name || "");
  const out = [];
  if (/^\d+$/.test(n) || /^layer_\d+$/.test(n)) out.push("This app sends its arrays without names (or names are hidden with <code>viz-layer-names = false</code>), so they are just numbered in order.");
  if (/conv/i.test(n)) out.push("<b>conv</b> = a convolution layer: small filters that slide over an image and detect patterns such as edges or textures.");
  if (/fc|linear|dense|classifier|head/i.test(n)) out.push("<b>fc / linear</b> = a fully connected layer: every input is wired to every output. Usually where the final decision is made.");
  if (/bn|norm/i.test(n)) out.push("<b>norm</b> = a normalisation layer that keeps the numbers flowing through the network in a healthy range.");
  if (/emb/i.test(n)) out.push("<b>embedding</b> = a lookup table that turns words or ids into vectors.");
  if (/attn|attention|q_proj|k_proj|v_proj/i.test(n)) out.push("<b>attention</b> = the part of a transformer that decides which inputs to focus on.");
  if (/weight$/i.test(n)) out.push("<b>weight</b> = the strengths of the connections in this layer.");
  if (/bias$/i.test(n)) out.push("<b>bias</b> = one offset per output, a small extra knob added after the weights.");
  if (/running_mean|running_var|num_batches/i.test(n)) out.push("These are running statistics, not trained by gradient descent; they still travel with the model.");
  return out.join(" ");
}

const STATES = {
  idle: "Connected but not part of what is happening right now. In this round the server did not pick it, or it has finished its job.",
  selected: "The server picked this client for the current round and sent it a message. It is waiting for the message to arrive or about to start.",
  training: "Learning on its own private data, starting from the global model it just received. The data never leaves this client.",
  evaluating: "Testing the current global model on its own data, to see how good the model is here. Nothing is learned in this step.",
  working: "Running a custom job the app defined: not plain training or testing. Examples are computing local statistics or one step of local clustering for K-means. The message type says which job.",
  sending: "Finished. Its reply (an updated model, or test scores) is travelling back to the server.",
  done: "The server has received this client's reply for this round.",
  failed: "Something went wrong: the client's code raised an error, it disconnected, or it did not reply in time. The server carries on without it.",
  offline: "No longer connected to the federation.",
};

const SERVER_STATUS = {
  idle: "Not waiting for anyone and not merging. Between rounds, or the run has ended.",
  waiting: "It has sent the model out and is waiting for the chosen clients to reply.",
  aggregating: "Merging the clients' replies into one new global model (or one combined score). This is the moment federated learning happens.",
};

const STRATEGIES = {
  FedAvg: "Federated Averaging: the new global model is the average of the clients' models, weighted by how many examples each client trained on. The classic, simplest strategy.",
  FedProx: "Like FedAvg, but each client is gently pulled back towards the global model while it trains (a 'proximal' term). This helps when clients' data are very different from each other.",
  FedMedian: "For every single number in the model, take the median of the clients' values instead of the average. One wild client can't drag the result far, so it is robust to outliers.",
  FedTrimmedAvg: "Drop the most extreme client values at both ends for every number, then average the rest. Robust to a few bad clients.",
  Krum: "Pick the one client update that is closest to most other updates and use it. Defends against malicious clients.",
  MultiKrum: "Like Krum, but averages several of the most 'central' client updates.",
  Bulyan: "A combination of Krum-style selection and trimmed averaging. Strong defence against malicious clients.",
  FedAdam: "Average the client updates, then apply them on the server with the Adam optimiser (which adapts step sizes). Often converges faster.",
  FedYogi: "Like FedAdam, with a different, more stable rule for adapting step sizes.",
  FedAdagrad: "Average the client updates, then apply them on the server with the Adagrad optimiser.",
  FedAvgM: "FedAvg plus server-side momentum: the server keeps some of the previous round's direction, which smooths training.",
  QFedAvg: "A fairness-oriented FedAvg that gives more weight to clients where the model currently does badly.",
  "ServerApp without a Strategy": "This app runs its own loop instead of a Flower Strategy: it sends its own kinds of messages (shown as the phases of each round) and combines the answers in its own code. flwr-viz sees every message and reply, but not the combining step. A new round is counted each time the app repeats a kind of message.",
};

const LEVELS = {
  off: "Nothing is sent to flwr-viz.",
  minimal: "Only timing, message flow and errors are shared. No metrics and no per-layer numbers.",
  standard: "Adds the metrics your app returns and, per layer, its size and how much it changed. Never the weights themselves.",
  detailed: "Also adds per-layer mean, spread, minimum and maximum. Still never the weights themselves.",
};

const CHARTS = {
  "local train metrics": "What this client reported after training on its own data, round by round.",
  "local evaluation": "How well the global model it received did on this client's own test data, round by round.",
  "custom job": "Numbers this client sent back for one of the app's own jobs (not plain training or testing), round by round. What they mean is defined by the app's code.",
  "handler duration (s)": "How many seconds this client spent on each job (training or evaluating), per round. Spikes can mean a slow device or more data.",
  "live training loss (report_progress)": "The loss after each batch of the latest local training, as the client reported it. A downward slope means it is learning right now.",
  "aggregated train metrics": "The clients' training metrics combined by the server (averaged, weighted by number of examples), one point per round.",
  "aggregated client evaluation": "The clients' test results on their own data, combined by the server. This is how good the global model is <i>where the data lives</i>.",
  "central evaluation (server-side)": "The server tested the global model itself on a dataset it holds. Round 0 is the untrained starting model.",
  "global model change per round": "How far the global model moved in each round (the size of the merged update). It usually shrinks as training settles down.",
};

function metric(name, value, ctx) {
  const n = String(name);
  let simple;
  if (/num.?examples/i.test(n)) simple = "How many data points were used. The server uses it as a weight: a client with more data counts more in the average.";
  else if (/logloss|log_loss/i.test(n)) simple = "Log-loss: how surprised the model is by the right answers. 0 would mean perfectly confident and correct. Lower is better.";
  else if (/loss/i.test(n)) simple = "Loss: one number for how wrong the model is, averaged over the data. Lower is better. It is what training tries to push down.";
  else if (/acc/i.test(n)) simple = "Accuracy: the share of examples the model got right. 1.0 means all of them.";
  else if (/_s$|duration|time/i.test(n)) simple = "A duration in seconds.";
  else simple = "A number your app's code computed and put in its reply. flwr-viz only shows it; the meaning comes from your code.";
  let here = "";
  if (num(value) !== null) {
    if (/acc/i.test(n) && +value <= 1) here = `${f(value)} means about ${Math.round(+value * 100)} out of every 100 examples were right.`;
    else if (/num.?examples/i.test(n)) here = `${f(value, 6)} examples.`;
    else here = `Value: ${f(value)}.`;
    here += trend(n, value, ctx.prev);
  }
  const phase = ctx.phase === "train" ? " Measured on the data it trained on, so it can look better than reality."
    : ctx.phase === "evaluate" ? " Measured on held-out test data the model did not train on, which makes it a fairer score." : "";
  return { title: n, simple: simple + phase, here };
}

export function explain(key, c = {}) {
  switch (key) {
    case "client.title": return { title: "A client", simple: "One participant in the federation: a phone, hospital, or (in simulation) a process. It keeps its own data and only ever sends model updates and scores.", here: `Its full id is ${c.id}.` };
    case "state": return { title: `State: ${c.state}`, simple: STATES[c.state] || "", here: num(c.since) !== null ? `It has been in this state for ${(+c.since).toFixed(1)} s.` : "", tip: "The shirt colour of the character always shows this state." };
    case "client.id": return { title: "Node id", simple: "The number Flower uses to address this client. It is random and says nothing about the data.", here: "" };
    case "client.state": return { title: "State and last message", simple: STATES[c.state] || "", here: c.mtype ? `The last job it got was a <b>${c.mtype}</b> message: ${/train/.test(c.mtype) ? "learn on your data and send back an improved model" : /evaluate/.test(c.mtype) ? "test the model on your data and send back scores" : "a custom action defined by the app"}.` : "" };
    case "client.rounds": return { title: "Rounds", simple: "How many times the server chose this client. Servers often pick only a fraction of clients per round, so a client can sit some rounds out.", here: `It trained in ${c.tr} round${c.tr == 1 ? "" : "s"} and evaluated in ${c.ev}.` };
    case "client.messages": return { title: "Messages", simple: "Each round the server sends a message to every client it picks; the client answers with one reply.", here: `${c.msgs} received${c.failures ? `, ${c.failures} of which failed` : ""}${c.timeouts ? ` (${c.timeouts} timed out: the server stopped waiting)` : ""}.` };
    case "client.traffic": return { title: "Traffic", simple: "How much data went over the network. Federated learning sends model updates, never raw data, so this is mostly the size of the model times the number of rounds.", here: `${bytes(c.in)} came in, ${bytes(c.out)} went out.${num(c.out) === 0 ? " Nothing went out yet, or its replies were only small scores." : ""}` };
    case "client.time": return { title: "Time per job", simple: "Average seconds this client spent per training job and per evaluation job. Slow clients hold up a round, because the server waits for replies.", here: c.tr !== null && c.tr !== undefined ? `Training takes about ${f(c.tr)} s here${c.ev !== null && c.ev !== undefined ? `, evaluating about ${f(c.ev)} s` : ""}.` : "" };
    case "client.progress": return { title: "Live progress", simple: "Reported from inside the training loop with <code>report_progress()</code>. An <b>epoch</b> is one full pass over the client's data; a <b>batch</b> is a small chunk of it processed in one step.", here: `Epoch ${c.epoch}, batch ${c.batch}${c.total ? ` of ${c.total} (${Math.round(100 * c.batch / c.total)}% through this epoch)` : ""}${num(c.loss) !== null ? `, current loss ${f(c.loss)}` : ""}.` };
    case "client.error": return { title: "Last error", simple: "What went wrong the last time this client failed. Only the error type and first line are shared, never a full traceback or data.", here: c.err ? `<i>${c.err}</i>` : "", tip: "A failed client simply drops out of that round; the server aggregates whatever replies it did get." };
    case "client.blackbox": return { title: "Why no live progress?", simple: "flwr-viz watches messages going in and out of the client. It cannot see inside the training code unless that code reports its progress.", here: "Add one line, <code>report_progress(epoch=…, batch=…, loss=…)</code>, in the training loop to get a live loss curve here." };
    case "metrics.section": return { title: `Metrics from round ${c.round}`, simple: "The numbers this client sent back in its replies for its most recent round. <b>train</b> rows come from training, <b>evaluate</b> rows from testing.", here: "" };
    case "metric": return metric(c.name, c.value, c);
    case "chart": return { title: c.title, simple: CHARTS[c.kind] || "A number tracked over rounds.", here: c.name ? metric(c.name, c.last, { prev: c.prev, phase: c.phase }).simple : "" };
    case "layers.sent": return { title: "Update sent", simple: "The model this client sent back after training, layer by layer. The change columns compare it with what it received, which is exactly what this client learned.", here: "", tip: "Only summary numbers are shown. The weights themselves stay private to the federation." };
    case "layers.recv": return { title: "Model received", simple: "The global model the server sent to this client at the start of its latest job, layer by layer.", here: "" };
    case "layers.global": return { title: "Global model", simple: "The shared model the server keeps, after merging this round's client updates. 'Δ this round' is how much each layer changed in the merge.", here: "" };
    case "col.layer": return { title: "Layer", simple: "A model is a list of named arrays of numbers. Each row is one of them; each is drawn as one book in the scene." };
    case "col.params": return { title: "Params", simple: "How many numbers (parameters) this layer holds. Bigger layers are drawn as wider books." };
    case "col.norm": return { title: "‖w‖ (size)", simple: "The overall size of all the numbers in the layer together: imagine them as one long arrow; this is its length. Useful to compare with the change." };
    case "col.delta": return { title: c.label || "Δ (change)", simple: c.which === "global" ? "How far the global model's layer moved in this round's merge: the length of the arrow from the old numbers to the new ones." : "How far this client moved the layer while training: the length of the arrow from the numbers it received to the numbers it sent." };
    case "col.rel": return { title: "rel (relative change)", simple: "The change divided by the size: Δ / ‖w‖. It makes big and small layers comparable. This is the colour of each book: pale blue is tiny, red is huge." };
    case "layer": {
      const rel = num(c.rel);
      const who = c.which === "global" ? "the merge this round" : c.which === "sent" ? "this client's training" : "the latest update";
      return {
        title: c.name, simple: layerKind(c.name) || "One block of the model's numbers.",
        here: `${num(c.params) !== null ? `${(+c.params).toLocaleString()} numbers. ` : ""}${num(c.norm) !== null ? `Size ${f(c.norm)}. ` : ""}${num(c.delta) !== null ? `Moved by ${f(c.delta)} in ${who}, i.e. ${pct(rel)} of its size: ${sizeWord(rel)}.` : c.which === "recv" ? "This is what arrived; the change is shown in the 'update sent' table." : "No change information yet."}`,
      };
    }
    case "server.title": return { title: "The server (the diamond)", simple: "The coordinator, drawn as the diamond in the centre. It knows everything the clients <i>send</i>: their model updates and scores. It never sees their data. Each round it picks clients, sends them the current model, and merges what comes back.", tip: "It spins faster and glows while it merges, and flashes when a new global model is ready." };
    case "server.status": return { title: `Server: ${c.status}`, simple: SERVER_STATUS[c.status] || "", here: "" };
    case "strategy": return { title: `Strategy: ${c.name}`, simple: STRATEGIES[c.name] || (/DifferentialPrivacy/.test(c.name || "") ? "Differential privacy wrapper: clips each client update and adds calibrated noise, so no single client's data can be read back from the global model." : "The rule the server uses to pick clients and merge their replies. This is a custom or less common strategy."), tip: "flwr-viz works with any strategy; it watches the strategy from outside without changing it." };
    case "server.round": return { title: "Round", simple: "One round = the server sends the model out, chosen clients train, the server merges their updates. Then (usually) clients test the new model.", here: `Round ${c.round} of ${c.total}.` };
    case "server.nodes": return { title: "Nodes seen", simple: "How many different clients have appeared in this run so far.", here: `${c.n} clients.` };
    case "server.messages": return { title: "Messages", simple: "Every job is a message from the server and a reply from a client. A failed reply means the client errored, vanished, or was too slow.", here: `${c.sent} sent, ${c.replies} replies came back, ${c.failed} failed.` };
    case "level": return { title: `Detail level: ${c.level}`, simple: LEVELS[c.level] || "", here: "", tip: "Set with <code>viz-level</code> in the app's run config." };
    case "replies": return { title: "Replies per round", simple: "For each round: how many chosen clients answered successfully, out of how many were chosen. A ✕ counts failures.", here: "" };
    case "replies.row": return { title: `Round ${c.round}`, simple: "Successful replies / clients chosen, for each kind of job in this round (training, evaluation, or the app's own message types).", here: `${c.cells}.${c.failed ? ` ${c.failed} failed; the server merged only the successful ones.` : ""}` };
    default: return null;
  }
}
