# flwr-viz — design plan (draft for review)

Target: **Flower 1.44.0** (latest on PyPI as of 2026-11-21, installed in `.venv`).
Everything below was checked against the installed source, not memory. File references are
relative to `site-packages/flwr/`.

## 0. What the real API looks like (findings)

| Fact | Where | Consequence for us |
|---|---|---|
| `Mod = Callable[[Message, Context, ClientAppCallable], Message]` | `clientapp/typing.py` | Our client mod is a plain function with this signature. |
| `ClientApp(mods=[...])` mods are bound when `@app.train()` etc. decorate (`app._mods + mods`) | `clientapp/client_app.py:426` | `ClientApp(mods=[viz_mod])` covers train, evaluate, query and custom actions. First in list = outermost, so it sees the reply *after* other mods (e.g. LocalDpMod) changed it — i.e. what actually leaves the node. |
| `Context` has `run_id`, `node_id`, `node_config`, `run_config`, `state` | `app/message/context.py` | Node id and run id come from the context; the collector address can come from `run_config`. |
| `Strategy` is an ABC with `configure_train/aggregate_train/configure_evaluate/aggregate_evaluate/summary` plus a concrete `start()` that drives the rounds | `serverapp/strategy/strategy.py` | We can wrap any strategy by instrumenting those methods and the `Grid` it is given. |
| `start()` calls the **module-level** `send_and_receive(grid, msgs, timeout)` which uses `grid.push_messages` / `grid.pull_messages` / `grid.get_node_ids` | `serverapp/strategy/strategy_utils.py:199` | A `Grid` proxy sees each push and each reply **as it arrives**, giving real per-message timing. |
| On deadline, `send_and_receive` **silently drops** missing replies; only nodes that vanish get a synthetic `Error(NODE_UNAVAILABLE)` reply | same | Timeouts are not reported by Flower. We infer them: sent message ids with no reply by the time `aggregate_*` is called ⇒ `msg.timeout`. |
| FedAvg builds messages without `group_id`; the round is only in `config["server-round"]` (a convention of the built-in strategies) | `serverapp/strategy/fedavg.py:154,180` | Clients can't reliably know the round. Correlate by **message id** instead: the server emits `msg.sent{msg_id, round}`, the client emits `client.recv{msg_id}`; the collector fills in `round` on client events. No message mutation. |
| ClientApps in simulation run in Ray actor **processes**; the ServerApp runs in its own process | `server/superlink/fleet/vce/backend/raybackend.py` | Every process has its own emitter. Events always go over the network (localhost in simulation). |
| `flwr new` in 1.44 no longer bundles templates; it downloads from Flower Hub (`@flwrlabs/quickstart-pytorch`, `-tensorflow`, `-sklearn`, `-numpy`, `-mlx`, `-xgboost`) | `cli/new/new.py` | Example 1 = `@flwrlabs/quickstart-pytorch` (downloaded into `_template_ref/`). |
| `Array` holds serialized bytes + `dtype`/`shape`; `.numpy()` deserializes. XGBoost apps ship one bytes blob, not tensors. | `app/message/array.py` | Layer stats are computed only for numeric arrays; blobs are reported as `{name, bytes}` with no norm. |

## 1. Package layout

```
flwr-viz/
  pyproject.toml            # package "flwr-viz", depends on flwr>=1.37, numpy; extra [collector] = aiohttp
  src/flwr_viz/
    schema.py               # event constructors, SCHEMA_VERSION = 1, validation
    emitter.py              # non-blocking background sender (shared by server + clients)
    stats.py                # per-layer summaries (privacy levels)
    mod.py                  # viz_mod / make_viz_mod(...)
    progress.py             # report_progress(...) helper for training loops
    strategy.py             # instrument(strategy, ...) / VizStrategy wrapper
    collector/server.py     # aiohttp: POST /ingest, GET /ws, static frontend, JSONL recorder
    collector/static/       # three.js frontend (three vendored, no build step, works offline)
    cli.py                  # `flwr-viz serve`, `flwr-viz replay run.jsonl`
  tests/                    # pytest: mod, strategy wrapper, collector, unreachable collector, replay
  examples/
    pytorch-fedavg/         # from `flwr new @flwrlabs/quickstart-pytorch` + 3 lines
    sklearn-fedprox/  (or numpy + FedAdam/FedMedian; see open questions)
  docs/SCHEMA.md
  README.md
```

## 2. Enabling it (the "few lines")

```python
# client_app.py
from flwr_viz import viz_mod
app = ClientApp(mods=[viz_mod])

# server_app.py
from flwr_viz import instrument
strategy = instrument(FedAvg(fraction_evaluate=0.5))   # same object, any Strategy
result = strategy.start(grid=grid, ...)                # unchanged call

# optional, inside the training loop
from flwr_viz import report_progress
report_progress(epoch=e, batch=i, total_batches=n, loss=loss.item())
```
```bash
flwr-viz serve            # http://127.0.0.1:7878, records to ./flwr-viz-runs/<run>.jsonl
```
Collector address: `viz-url` in the app's `[tool.flwr.app.config]` (travels with the run to every
node, including real deployments) → else env `FLWR_VIZ_URL` → else `http://127.0.0.1:7878`.
Detail level: `viz-level` the same way, or `make_viz_mod(level=...)` / `instrument(..., level=...)`.

## 3. Components

### 3.1 Emitter (shared)
- `emit(event)` = build dict + `queue.put_nowait`. Never blocks, never raises (all errors swallowed).
- Bounded queue (default 10 000). When full, the new event is dropped and a counter goes up.
- One daemon thread batches events (≤ 200 or every 100 ms) and POSTs JSON to `/ingest` with a
  0.5 s timeout, stdlib `urllib` only (no extra deps in the training process).
- Circuit breaker: after a failed POST, drop batches for 2 s, then 5 s, up to 30 s, before retrying.
  If the collector is down, cost per event is one dict + one queue op.
- `atexit` flush with a 1 s cap, so short-lived ClientApp processes (deployment isolation mode)
  still deliver their last events. Dropped-event counts are sent in the next successful batch
  (`sys.dropped`) so the UI can say "N events lost".
- `FLWR_VIZ_DISABLE=1` turns everything into no-ops.

### 3.2 Client mod
Per message: `client.recv` → `client.start` → (user handler, with progress events) → `client.end`.
Exceptions from the handler are caught only to emit `client.end{ok:false, error}` and then
**re-raised unchanged**, so Flower's own error handling is untouched.
The mod puts `{node_id, msg_id, mtype}` in a `contextvars.ContextVar` so `report_progress` knows
where it is without the user passing anything. `report_progress` is rate-limited (default ≤ 5/s
per node; first and last batch of each epoch always sent) and does nothing outside a mod.

### 3.3 Strategy wrapper
`instrument(strategy)` wraps, on that instance: `configure_train/evaluate`, `aggregate_train/evaluate`,
and `start` (to swap in a `Grid` proxy and wrap `evaluate_fn`). Returns the same object, so
`isinstance`, attributes and custom methods keep working. Works for FedAvg, FedProx, FedAdam, Krum,
DP wrappers, and custom subclasses — anything that goes through `configure_*`/`aggregate_*` and the grid.

| Event | Source |
|---|---|
| `run.start` | `start()` entry: strategy class name, `num_rounds`, timeout, initial model layer summary |
| `round.start` (phase train/evaluate) | `configure_*` call: `connected` = `grid.get_node_ids()`, `sampled` = dst ids of returned messages |
| `msg.sent` | grid proxy `push_messages`: msg id, dst node, type, bytes, round |
| `msg.reply` | grid proxy `pull_messages` (real arrival time) + synthetic error replies seen in `aggregate_*`: ok / error code + reason, latency |
| `msg.timeout` | at `aggregate_*`: sent ids with no reply |
| `agg.start` / `agg.end` | around `aggregate_*`: replies ok/failed, duration, aggregated metrics, per-layer global change |
| `eval.central` | wrapped `evaluate_fn` |
| `round.end`, `run.end` / `run.error` | after evaluate phase; `start()` return / exception (re-raised) |

The global model per-layer change is computed on the server, which already holds the global
model; only norms are emitted.

### 3.4 Collector
aiohttp app (one process, one port):
- `POST /ingest` — accepts a batch, stamps `seq` (global order) and `rts` (receive time), fills
  `round` on client events from the `msg_id → round` map, appends to JSONL, broadcasts.
- `GET /ws` — on connect, sends history of the current run (bounded, default last 200k events),
  then live events.
- `GET /` — frontend. `GET /api/runs`, `GET /api/runs/<file>` for recorded runs.
- `flwr-viz replay run.jsonl` serves a recorded file with no federation; the frontend can also
  open a dropped `.jsonl` file directly.
- Binds `127.0.0.1` by default. For real deployments `--host 0.0.0.0` plus optional
  `--token` (shared secret sent as a header by emitters). No TLS of its own — document running
  behind a reverse proxy if needed.

### 3.5 Frontend (three.js, vendored)
- **State is a pure reducer over events.** Live mode follows the head; the timeline scrubs by
  replaying from per-round snapshots (fast seek even for long runs). Pause / play / speed
  (0.25×–8×) / jump to round.
- Scene: server at centre with its layer stack; clients on a ring (rings / Fibonacci sphere past
  ~60). Client state colour + glyph: idle, selected, training, sending, failed, timed-out.
- Messages: instanced particles along arcs, server→client on `msg.sent`/`client.recv`,
  client→server on `client.end`/`msg.reply`; red burst for failures.
- Layer stacks: one block per layer, height ∝ log(param count), colour ∝ relative change
  (`delta_norm / norm`), shared colour scale with legend.
- Aggregation: returned client stacks fly to the server and merge into the global stack, which
  then recolours by global change.
- Detail panel on click: node info, per-round metric charts (canvas, no chart lib), last
  messages, per-layer table, progress (loss curve) if the helper is used.
- Scale: clients are one `InstancedMesh`; particles one instanced pool. Past ~40 clients, layer
  stacks are drawn only for the server and the selected/hovered client; past ~150, particles are
  sampled (aggregate "flow" counts kept). Target: 60 fps at 10 nodes, ≥ 30 fps at 300 on a laptop.

## 4. Event schema v1 (full spec goes in `docs/SCHEMA.md`)

Envelope (every event):
```json
{
  "v": 1,
  "type": "client.end",
  "ts": 1763712000.123,          // sender wall clock, seconds
  "run_id": 4821,
  "src": {"role": "client", "node_id": 1234567},
  "round": 3,                    // null if unknown at emit time; collector fills it from msg_id
  "msg_id": "ab12…",             // when the event concerns one message
  "data": { ... }                // per-type payload below
}
```
Added by the collector: `seq` (int, total order), `rts` (collector receive time).

Layer summary object (`LayerStat`): `{"name","shape","dtype","params","bytes","norm","delta_norm"?,
"mean"?,"std"?,"min"?,"max"?}` — which fields appear depends on the level (below). Arrays that
are not numeric give only `name`, `dtype`, `bytes`. Models with more than `max_layers` (default 64)
arrays: the rest are merged into one `"(other N arrays)"` entry.

| type | role | data |
|---|---|---|
| `run.start` | server | `strategy`, `num_rounds`, `timeout`, `level`, `layers:[LayerStat]` |
| `round.start` | server | `phase` (`train`/`evaluate`), `connected:[id]`, `sampled:[id]` |
| `msg.sent` | server | `dst`, `mtype`, `bytes` |
| `msg.reply` | server | `from`, `ok`, `error_code?`, `error_reason?`, `latency_s`, `bytes` |
| `msg.timeout` | server | `dst`, `waited_s` |
| `agg.start` | server | `phase`, `n_replies`, `n_ok`, `n_failed` |
| `agg.end` | server | `phase`, `duration_s`, `metrics{}`, `layers:[LayerStat]` (train only; delta = change in global model) |
| `eval.central` | server | `metrics{}` |
| `round.end` | server | `duration_s` |
| `run.end` / `run.error` | server | `duration_s`, `rounds_completed` / `error` |
| `client.recv` | client | `mtype`, `bytes`, `layers:[LayerStat]` (received model) |
| `client.start` | client | `mtype` |
| `client.progress` | client | `epoch`, `batch`, `total_batches?`, `loss?`, `extra{}` (numbers only) |
| `client.end` | client | `mtype`, `ok`, `duration_s`, `error?`, `metrics{}`, `bytes`, `layers:[LayerStat]` (with `delta_norm` = ‖returned − received‖ per layer) |
| `sys.dropped` | any | `count` |

Metrics: only `MetricRecord` scalars/lists of numbers are copied; `ConfigRecord` contents are never
sent (they may carry anything).

### Privacy / detail levels (`viz-level`)
| level | what leaves the process |
|---|---|
| `off` | nothing |
| `minimal` | timing, message flow, states, errors, byte counts. No metrics, no layer info. |
| `standard` (default) | + metrics + per-layer name/shape/params/norm/delta_norm |
| `detailed` | + per-layer mean/std/min/max |

Never emitted at any level: raw weights, gradients, training data, labels, config values,
`node_config` (e.g. partition ids), error tracebacks (only exception type + first line of message,
truncated). Layer names can be turned off (`viz-layer-names=false` → `layer_0…`) for architectures
someone considers sensitive.

## 5. Limits on what can be observed (will be in the README)
1. **Inside local training is invisible** without `report_progress`. The mod sees one handler call
   with a duration; epochs/batches/loss only appear if the user reports them.
2. **"Sent" means pushed to the SuperLink**, not delivered. Real delivery is visible only as the
   client's `client.recv` (if that client's events reach the collector). In simulation there is no
   network hop at all, so travel animation is illustrative timing, not measured wire time.
3. **Timeouts are inferred** at aggregation time, because Flower discards them silently.
4. **Clock skew between machines**: `ts` is the sender's clock. Ordering uses collector `seq`;
   durations use same-clock differences only (client durations from client clock, latencies from
   server clock). No cross-machine latency is computed.
5. **Strategies that bypass `configure_*`/`aggregate_*`** (custom `start()` doing raw grid calls)
   still get grid-level events (sent/reply/timeout per message) but no round/aggregation events.
6. **What the server aggregates is what the strategy gets.** Secure aggregation / server-side DP
   change things in ways we only see at the boundaries.
7. **Mod order matters**: put `viz_mod` first to see what leaves the node; later mods' changes
   (e.g. DP noise) are then included in `delta_norm`.
8. Non-tensor payloads (XGBoost model bytes) get byte counts, not norms.
9. Computing norms costs one pass over the parameters per message (milliseconds for millions
   of parameters; measured and reported in tests). `minimal` skips it entirely.
10. Deployment mode with real SuperNodes is designed for (network address via run config) but I
    will only test simulation and a local multi-process deployment if time allows.

## 6. Verification plan
1. `pytest`: mod (events, ordering, delta norm correctness on a toy model, exception passthrough,
   privacy levels never emit arrays), strategy wrapper (with a fake Grid: sampled nodes, replies,
   synthetic NODE_UNAVAILABLE, inferred timeouts, global layer delta, `run.error` re-raise),
   collector (ingest → JSONL → WS broadcast, round fill-in), **unreachable collector** (training
   overhead bounded, no exceptions, drop counter), replay of a recorded JSONL with no federation.
2. Example 1: `@flwrlabs/quickstart-pytorch` + plug-in, `flwr run . --stream` in simulation with
   10 nodes, collector running; check JSONL contents and open the scene in a real browser,
   screenshots of live and replay.
3. Example 2: different framework + different strategy (see question below), same checks.
4. Scale check: synthetic event generator for 300 clients to measure frontend fps.

## 7. Open questions for you
- **Second example**: my pick is `@flwrlabs/quickstart-sklearn` (logistic regression, no tensors
  framework at all) with **FedProx** swapped for… FedProx needs a proximal term in client code, so a
  cleaner proof is sklearn + **FedAdam** or **FedMedian**, which need no client change. Alternative:
  TensorFlow/Keras + FedProx. I'd go with sklearn + FedMedian unless you prefer otherwise.
- **Collector dependency**: aiohttp (one dependency, does HTTP + WebSocket + static). OK?
