// Replays a recorded JSONL through the frontend reducer (the same module the
// browser uses) and checks the result against facts recomputed from the raw file.
// Usage: node tests/js/replay_check.mjs run.jsonl   -> prints JSON, exit 1 on failure
import { readFileSync } from "node:fs";
import { Timeline } from "../../src/flwr_viz/collector/static/js/state.js";

const file = process.argv[2];
const evs = readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l));
evs.sort((a, b) => a.seq - b.seq);
const failures = [];
const check = (cond, msg) => { if (!cond) failures.push(msg); };

const tl = new Timeline(100);
for (const e of evs) tl.push(e);

// 1) times never go backwards
for (let i = 1; i < tl.times.length; i++) if (tl.times[i] < tl.times[i - 1]) { failures.push("time went backwards"); break; }

// 2) full replay matches the raw file
const end = tl.seek(tl.t1);
const sent = evs.filter(e => e.type === "msg.sent");
const replies = evs.filter(e => e.type === "msg.reply");
check(end.counts.events === evs.length, `applied ${end.counts.events} of ${evs.length}`);
check(end.counts.sent === sent.length, "sent count");
check(end.counts.replies === replies.length, "reply count");
const runEnd = evs.find(e => e.type === "run.end");
if (runEnd) {
  check(end.run.ended !== null, "run should be ended");
  check(end.run.round === runEnd.data.rounds_completed, `round ${end.run.round} vs ${runEnd.data.rounds_completed}`);
}
const perNode = {};
for (const e of sent) perNode[e.data.dst] = (perNode[e.data.dst] || 0) + 1;
for (const [id, n] of Object.entries(perNode)) check(end.nodes[id] && end.nodes[id].msgs === n, `node ${id} msgs`);
const clientIds = new Set(evs.filter(e => e.src.role === "client").map(e => String(e.src.node_id)));
for (const id of clientIds) check(String(id) in end.nodes, `client ${id} missing`);
const trainRounds = new Set(evs.filter(e => e.type === "agg.end" && e.data.phase === "train").map(e => e.round));
check(Object.keys(end.server.metrics.train).length === [...trainRounds].filter(r => evs.find(e => e.type === "agg.end" && e.round === r && Object.keys(e.data.metrics || {}).length)).length, "train metric rounds");

// 3) every client event got a round (collector correlation)
const noRound = evs.filter(e => e.src.role === "client" && e.round === null).length;
check(noRound === 0, `${noRound} client events without round`);

// 4) scrubbing: seeking back and forth gives the same state as a fresh replay
const marks = tl.roundMarks();
const probes = [tl.t0, ...marks.map(m => m.t), (tl.t0 + tl.t1) / 2, tl.t1];
const snapshot = s => JSON.stringify({ run: s.run, counts: s.counts, nodes: Object.fromEntries(Object.entries(s.nodes).map(([k, n]) => [k, [n.state, n.msgs, n.failures]])), sv: s.server.status });
for (const t of [...probes].reverse().concat(probes)) {
  const fresh = new Timeline(1e9); for (const e of evs) fresh.push(e);
  const a = snapshot(tl.seek(t)), b = snapshot(fresh.seek(t));
  if (a !== b) { failures.push(`seek mismatch at t=${t - tl.t0}`); break; }
}

console.log(JSON.stringify({
  file, events: evs.length, rounds: marks.length, nodes: end.order.length, sent: sent.length,
  failed: end.counts.failed, timeouts: end.counts.timeouts, finished: end.run.ended !== null,
  feedTail: end.feed.slice(-3).map(f => f.text), failures,
}, null, 1));
process.exit(failures.length ? 1 : 0);
