// Plan the learn-mode tour for a recording and check every question can be answered correctly.
// Usage: node tests/js/tour_check.mjs run.jsonl  -> JSON summary, exit 1 on failure
import { readFileSync } from "node:fs";
import { Timeline } from "../../src/flwr_viz/collector/static/js/state.js";
import { planTour, buildSteps } from "../../src/flwr_viz/collector/static/js/tour.js";

const file = process.argv[2];
const evs = readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l)).sort((a, b) => a.seq - b.seq);
const tl = new Timeline(); for (const e of evs) tl.push(e);
const P = planTour(tl), steps = buildSteps(P);
const failures = [];
const calls = [];
const api = new Proxy({}, { get: (_, k) => (...a) => calls.push(k) });   // records what each step asks the app to do
for (const s of steps) {
  s.setup?.(api);
  let ok;
  if (s.multi) ok = s.check(["updates", "metrics"]) && !s.check(["data", "updates", "metrics"]);
  else if (typeof s.pick === "string" && s.pick.startsWith("book:")) ok = s.check(s.pick.slice(5), P.bestLayer.index) && P.bestLayer.index < P.bestLayer.count;
  else if (Array.isArray(s.pick)) ok = s.pick.some(id => s.check(id)) && s.pick.filter(id => s.check(id)).length === 1;
  else if (s.pick === "any") ok = s.check(String(P.sampled?.[0] ?? "1")) && !s.check("server");
  else ok = (s.choices || []).filter(c => s.check(c.value)).length === 1;
  if (!ok) failures.push(`step ${s.id}: no single correct answer among its options`);
  const text = s.reveal(api, true);
  if (typeof text !== "string" || text.length < 20 || /undefined|NaN/.test(text)) failures.push(`step ${s.id}: bad reveal text: ${text}`);
  if (/undefined|NaN/.test(s.q)) failures.push(`step ${s.id}: bad question: ${s.q}`);
}
console.log(JSON.stringify({ file, strategy: P.strategy, round: P.round, steps: steps.map(s => `${s.level}:${s.id}`), failures,
  sample: steps.map(s => s.q).slice(2, 5) }, null, 1));
process.exit(failures.length ? 1 : 0);
