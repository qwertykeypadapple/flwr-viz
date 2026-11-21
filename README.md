# flwr-viz

**[See it running →](https://qwertykeypadapple.github.io/flwr-viz/)** a presentation with the viewer replaying real Flower runs, and a question-driven tour of federated learning.

A live 3D view of a [Flower](https://flower.ai) federated learning run in the
browser. The server sits in the middle and the clients around it, and you can
watch the model travel out and the updates come back, round by round and message
by message. It is a teaching and debugging tool. It works with any Flower app
(any model, dataset, ML framework or strategy) and is a separate package that
depends on `flwr`; Flower itself is not modified.

Built and tested against **Flower 1.44.0**, using the Message API: `ServerApp`,
`ClientApp`, `flwr.clientapp.mod` and `flwr.serverapp.strategy`. It requires
`flwr>=1.37` but has only been run on 1.44.

```
ClientApp + viz_mod ───┐  HTTP POST /ingest   ┌──────────────┐  WebSocket  ┌─────────────────┐
ClientApp + viz_mod ───┼────────────────────▶ │  collector   │ ──────────▶ │ browser (three) │
ServerApp + instrument ┘  batched, never      │  + JSONL     │             │  live / replay  │
                          blocks training     └──────────────┘             └─────────────────┘
```

## Install

It is not on PyPI yet. Clone it and install from the folder:

```bash
git clone https://github.com/qwertykeypadapple/flwr-viz.git
cd flwr-viz
pip install -e ".[collector]"
```

The training side needs only `flwr` and `numpy`. Only the machine that runs the
collector needs `aiohttp`.

## Enable it (three changes)

**1. Client** (`client_app.py`): add the mod. Put it first in the list so it
sees exactly what leaves the node, after any other mod (for example differential
privacy) has run.

```python
from flwr_viz import viz_mod
app = ClientApp(mods=[viz_mod])
```

**2. Server** (`server_app.py`): wrap your strategy. Any `Strategy` works:
FedAvg, FedProx, FedMedian, Krum, the DP wrappers, or your own subclass.
`instrument` returns the same object, so the rest of your code stays the same.

```python
from flwr_viz import instrument
strategy = instrument(FedAvg(fraction_evaluate=0.5), run_config=context.run_config)
result = strategy.start(grid=grid, initial_arrays=arrays, ...)
```

**Apps without a Strategy.** Some apps run their own loop and call
`grid.send_and_receive` directly, like the federated K-means app on Flower Hub.
For those, instrument the grid instead, as the first line of `main`:

```python
from flwr_viz import instrument

@app.main()
def main(grid: Grid, context: Context) -> None:
    grid = instrument(grid, run_config=context.run_config)
    ...
```

Every message and reply is then recorded, and timeouts are inferred. Rounds are
inferred too: a new round starts whenever the app sends a kind of message it
already sent in the current round. The app combines replies in its own code, so
there are no aggregation events or global-model layers. If the app also runs a
Strategy on the same grid, instrument both: the strategy's recorder takes over
while it runs, and nothing is counted twice.

**3. Start the collector, open the page, run your app as usual**

```bash
flwr-viz serve
```

The page is at http://127.0.0.1:7878. Then start your app with `flwr run . --stream`.

**Optional: progress from inside training.** A mod only sees where messages
start and end, so by default local training is a black box: the view shows
"training" and how long it took. One call in your training loop adds epochs,
batches and a live loss curve. Outside a ClientApp the call does nothing, so it
is safe to leave in.

```python
from flwr_viz import report_progress
for epoch in range(epochs):
    for i, batch in enumerate(loader):
        ...
        report_progress(epoch=epoch, batch=i, total_batches=len(loader), loss=loss.item())
```

### Configuration

Set these in your app's `[tool.flwr.app.config]`. Flower sends the run config
to every node, so this one place configures the server and all clients,
including clients on other machines. Environment variables and keyword arguments
work too. Precedence is argument, then run config, then environment variable,
then default.

| run config key | env var | default | meaning |
|---|---|---|---|
| `viz-url` | `FLWR_VIZ_URL` | `http://127.0.0.1:7878` | where to send events |
| `viz-level` | `FLWR_VIZ_LEVEL` | `standard` | `off`, `minimal`, `standard` or `detailed` (see Privacy) |
| `viz-token` | `FLWR_VIZ_TOKEN` | none | shared secret; must match `flwr-viz serve --token` |
| `viz-layer-names` | `FLWR_VIZ_LAYER_NAMES` | `true` | `false` sends `layer_0, layer_1, …` instead of real names |
| `viz-max-layers` | `FLWR_VIZ_MAX_LAYERS` | `64` | arrays beyond this number are merged into one entry |
| `viz-progress-hz` | `FLWR_VIZ_PROGRESS_HZ` | `5` | maximum `report_progress` events per second per node |
| | `FLWR_VIZ_DISABLE=1` | | turns everything off |

flwr-viz itself does not pin a Flower version. It only lists `flwr`: the app
already pins its own Flower, and a lower bound here broke `uv sync` for an app
that declares Python ≥3.10, because Flower ≥1.37 needs Python 3.11. Importing
flwr-viz warns if Flower is older than 1.37.

### Flower 1.44: per-run environments

`flwr run` now installs each app's dependencies into a fresh environment with
`uv sync`, resolving them from PyPI. Until `flwr-viz` is on PyPI, add this to
the app's `pyproject.toml`. The path must be absolute, because Flower copies the
app to `~/.flwr/apps/` before `uv` runs.

```toml
[tool.uv.sources]
flwr-viz = { path = "/absolute/path/to/flwr-viz", editable = true }
```

Keep `editable = true`. Without it, uv caches a built copy and later runs keep
using old code (this happened during testing). Also, the default simulation now
has only 2 SuperNodes. For a more interesting picture, pass
`--federation-config "num-supernodes=10"`.

### Real deployments (clients on other machines)

On a machine that every node can reach, run
`flwr-viz serve --host 0.0.0.0 --token <secret>`. Then set
`viz-url = "http://<that-host>:7878"` and `viz-token = "<secret>"` in the run
config. The collector speaks plain HTTP, so put it behind a TLS reverse proxy if
the network isn't trusted. This setup has not been tested; see "Not verified"
below.

## Using the view

The page is drawn in a light, hand-drawn comic style: toon shading, ink outlines,
warm paper colours and Patrick Hand lettering (the font is bundled, so it works offline).

- **Clients are characters** standing in a ring, facing the centre. Once there
  are more clients than fit on one ring, the scene becomes a tower. Floors stack
  downwards (labelled FLOOR 1, 2, …), each holding up to about 53 clients on two
  rings, and you can see the lower floors through the translucent ones above.
  Past 60 clients the characters switch to a lower-poly version.
- **The server is a blue diamond with a gold ring** floating over the centre of
  the top floor. It spins and glows while it merges updates, flashes when a new
  global model is ready, and turns red if the run fails. Paper planes fly
  between it and every floor.
- **Shirt colour is the node's state**, with a matching ring on the floor:
  - white: idle
  - blue: selected (waves at the server)
  - green: training (jogs on the spot, with a progress bar over the head if you use `report_progress`)
  - lavender: evaluating (hand on chin)
  - orange: sending (throws the update)
  - mint: done (hops)
  - red: failed (dazed, dizzy X eyes, a red ring spreads out)
  - grey: offline (asleep)
- **A model is a stack of books**, one book per layer, beside its owner. A
  book's width grows with the square root of the layer's parameter count. Its
  colour is how much the layer changed, as ‖Δ‖/‖w‖ on a log scale from 0.01%
  (pale blue) to 100% (red); beige means there is no change information. A
  client's Δ is what it sent minus what it received. The server's pile is the
  global model, and its Δ is the new global model minus the previous one.
- **Messages are paper planes.** Blue ones carry the global model out, orange
  ones carry updates back, and red ones are failures.
- **Aggregation.** The books of the clients that replied fly onto the server's
  pile, which then flashes and takes the colours of the new global change.
- **Click anyone (or the server button) to see a projection.** The camera glides
  over, the character turns to face you, and a beam of light rises from their
  head into a card with everything known about them:
  - state and how long they've been in it
  - rounds trained and evaluated, messages, failures and timeouts, traffic, average durations
  - live epoch, batch and loss
  - last error
  - latest metrics, and charts per round
  - per-layer tables of the update sent and the model received

  The server's projection shows aggregated and central metrics, replies per
  round, and the global model's layers. Press Esc or × to close.
- **Hover anything in a card for an explanation.** A second projection slides
  out sideways on a horizontal beam from the item. It explains the item in
  plain words, then says what this particular value means right now. Some
  examples:
  - a loss: is it up or down since the last round, and is that good?
  - an accuracy of 0.55: about 55 out of every 100 examples were right
  - a layer row: what a conv or fc layer is, how many numbers it holds, and
    whether its change is tiny, moderate or very large
  - a strategy name: how FedAvg, FedProx, FedMedian, Krum and the others merge updates
  - a state, a chart, a column header

  The text lives in `static/js/explain.js`.
- **Timeline controls:**
  - play/pause (space) and previous/next round (← →)
  - speed from 0.25× to 16×
  - a scrub bar with round markers
  - "skip gaps", which jumps over long quiet stretches
  - "go live"

  **Open recording** loads any saved run. You can also drop a `.jsonl` file on the page.
- **"What just happened"** describes the run in plain sentences.
- **Scale.** Every character part is instanced, about 80 draw calls for any
  number of clients. Past 40 clients, books are drawn only for the selected or
  hovered client. Past 150, only a sample of the paper planes is drawn. With 300
  clients on six floors, a frame takes about 7–12 ms to render on an Apple M3.

## Learn mode: a question-driven tour

Press **learn** in the header. The tour takes a newcomer through federated
learning from the top down, using whatever run is loaded: a live run, a
recording, or a Flower Hub app's replay. At each step the timeline pauses and
asks a question. You answer by clicking in the scene (a client, the diamond, a
book) or by picking a choice. Then that part of the run plays and is explained
with this run's real numbers.

| level | questions |
|---|---|
| 1. The federation | Where is the training data? (locked chests appear beside each client) · If data can't move, what travels? (the model's paper planes take off) |
| 2. A round | Does the server ask all N clients? · Which client will send its update back first? · A client goes silent: does the round wait forever? (uses a real failure if the run had one) |
| 3. One client | What does a client do with the model? · Which layer (book) did it change most? |
| 4. Merging | Whose update counts more: the client with more data? (the answer depends on the strategy, so FedMedian says "the same") · One client sends garbage: average or median? (drag the bad value on a number line and watch both) |
| 5. Privacy | What does the server actually see? (the lens greys out the data chests) |

How it behaves:
- A wrong answer gets a hint and a second try, never a red X.
- "show me" skips any question.
- The end screen shows your first-try score and a one-line summary per level.
- Steps a run can't support are left out or explained without a demo. For
  example, a run where every client failed gets a 6-step tour.
  `tests/js/tour_check.mjs` checks that every question planned from every
  recording has exactly one correct answer (this also runs in `pytest`).

## Recording and replay

The collector writes every event of every run to
`./flwr-viz-runs/run-<run_id>-<time>.jsonl`. Change the folder with
`--record-dir`, or turn recording off with `--no-record`. To replay a run later,
with no federation running:

```bash
flwr-viz replay flwr-viz-runs/run-<run_id>-<time>.jsonl
```

The replay is served at http://127.0.0.1:7878 by default. Add `--port` to change that.

## Privacy

Federated learning exists so that data and raw weights stay where they are.
flwr-viz sends only summary statistics, and you choose the level:

| level | what leaves each process |
|---|---|
| `off` | nothing |
| `minimal` | timing, message flow, states, errors, byte counts. No metrics, no layer stats, no loss |
| `standard` | adds the metrics your app already returns, plus per layer: name, shape, parameter count, L2 norm, and L2 norm of the change |
| `detailed` | adds per-layer mean, std, min and max |

At no level does it send:
- weights or gradients
- training data or labels
- `ConfigRecord` contents
- `node_config` (for example partition ids)
- tracebacks; only the exception type and the first line of its message, cut to 200 characters

A test checks this: `tests/test_mod.py::test_never_emits_raw_values_or_config`.

Summary statistics still carry some information. For example, the norm of a
client's update is a quantity that some membership-inference attacks use. For
sensitive deployments:
- use `minimal`
- turn off layer names
- set a token
- keep the collector on a trusted network

## Event schema

All parts share one versioned JSON schema, v1. The full specification is in
[docs/SCHEMA.md](docs/SCHEMA.md). A sample event:

```json
{"v": 1, "type": "client.end", "ts": 1795262067.87, "run_id": "8100…",
 "src": {"role": "client", "node_id": "2607…"}, "round": 3, "msg_id": "8f1c…",
 "data": {"mtype": "train", "ok": true, "duration_s": 8.77,
          "metrics": {"train_loss": 2.21, "num-examples": 4000}, "bytes": 249440,
          "layers": [{"name": "conv1.weight", "shape": [6,3,5,5], "params": 450,
                      "norm": 2.49, "delta_norm": 1.95}]}}
```

Event types:
- **Server:** `run.start`, `round.start`, `msg.sent`, `msg.reply`, `msg.timeout`, `agg.start`, `agg.end`, `eval.central`, `round.end`, `run.end`, `run.error`
- **Client:** `client.recv`, `client.start`, `client.progress`, `client.end`
- **Either:** `sys.dropped`

Flower's ids are 64-bit integers, which JavaScript would round, so **all ids are
strings**. The collector adds two fields, `seq` (the total order of arrival) and
`rts` (the receive time). It also fills in `round` on client events by matching
their `msg_id` to the server's `msg.sent`.

## What can and cannot be observed

1. **The inside of local training is invisible** to a mod. You get its duration
   and its result, plus whatever you send with `report_progress`.
2. **"Sent" means handed to the SuperLink**, not delivered. Delivery shows up as
   the client's `client.recv`. In simulation there is no network at all, so the
   travel animation is illustrative, not measured wire time.
3. **Timeouts are inferred.** When replies miss the deadline, Flower's
   `send_and_receive` drops them without reporting anything. flwr-viz compares
   what was sent with what reached `aggregate_*` and emits a `msg.timeout` for
   each missing reply. Nodes that disconnect show up as Flower's own
   `NODE_UNAVAILABLE` error replies.
4. **Clocks differ between machines.** Events are ordered by the collector's
   `seq`. Durations are always measured on a single clock: client durations on
   the client, latencies on the server. The timeline never runs backwards, even
   when timestamps do.
5. **Custom strategies that bypass `configure_*` / `aggregate_*`**, with their
   own `start()` that talks to the grid directly, still produce per-message
   events from the Grid proxy, but no round or aggregation events.
6. **Secure aggregation and server-side DP** change updates in ways that are
   visible only at the boundaries the wrapper sees.
7. **Non-tensor payloads**, such as XGBoost model bytes, get byte counts but no norms.
8. **Cost.** The norms take one pass over the parameters per message. That is
   about 7 ms per message for a 1M-parameter model on this Mac, measured in
   `tests/test_unreachable.py`. The `minimal` level skips this work. With the
   collector unreachable, 4 real runs of the sklearn example alternated between
   `viz-level=off` (59.5 s, 52.7 s) and a dead `viz-url` (55.7 s, 58.3 s). The
   difference is within run-to-run noise, and those runs logged no errors.

## Examples

Both examples were created with `flwr new` from Flower Hub. Each differs from
its original template only by the plug-in lines and the `[tool.uv.sources]`
block.

| example | framework | strategy | shows |
|---|---|---|---|
| [`examples/pytorch-fedavg`](examples/pytorch-fedavg) | PyTorch CNN on CIFAR-10 | FedAvg with central evaluation | named layers, `report_progress` loss curves |
| [`examples/sklearn-fedmedian`](examples/sklearn-fedmedian) | scikit-learn logistic regression on Iris | FedMedian, 60% sampling | unnamed arrays, no progress helper, `viz-level = "detailed"`, opt-in `fail-prob` to show client failures |

To run one, start the collector in one terminal:

```bash
flwr-viz serve
```

Then, in a second terminal, from the example's folder:

```bash
pip install -e . && flwr run . --stream --federation-config "num-supernodes=10"
```

Each example folder contains `recorded-run.jsonl` from a real run, which you can
replay straight away with `flwr-viz replay`.

## Tests

```bash
pip install -e ".[test]" && pytest
```

| file | covers |
|---|---|
| `test_mod.py` | <ul><li>event order</li><li>per-layer norms and Δ, checked against hand-computed values</li><li>metrics</li><li>that no raw values or config leak</li><li>every level, and hidden layer names</li><li>that exceptions are re-raised unchanged</li><li>error replies</li><li>progress rate limit and worker threads</li><li>non-tensor arrays</li><li>a real `ClientApp`</li><li>64-bit ids</li></ul> |
| `test_strategy.py` | <ul><li>a full FedAvg run on a fake Grid: sampling, matching sent messages to replies, global Δ checked against hand-computed values, central evaluation, round end and run end</li><li>inferred timeouts, `NODE_UNAVAILABLE` and crashed clients</li><li>`run.error` with the exception re-raised</li><li>FedMedian, FedProx and a custom strategy</li><li>wrapping twice</li><li>the `off` and `minimal` levels</li></ul> |
| `test_collector.py` | <ul><li>filling in rounds, and holding client events that arrive early</li><li>validation</li><li>one JSONL file per run</li><li>WebSocket history plus live events</li><li>token checks</li><li>path traversal</li><li>replay mode serving identical events, read-only</li><li>normalising integer ids</li></ul> |
| `test_unreachable.py` | <ul><li>refused connection, unroutable address, slow collector</li><li>full queue</li><li>recovery with a `sys.dropped` report</li><li>mod overhead with a dead collector</li></ul> |
| `test_replay_js.py` | <ul><li>replays recordings, including the two real Flower runs, through the browser's own reducer in Node</li><li>checks the counts against the raw file</li><li>checks that seeking back and forth gives the same state as a fresh replay</li></ul> |

For a UI load test without Flower, `flwr-viz synth --clients 300 --rounds 3`
streams synthetic events. The data is made up, so this tests rendering only.

## Status (2026-11-21)

**Verified**
- Both examples ran in Flower 1.44 simulation and streamed events live to the
  collector, with 0 events rejected:
  - PyTorch with FedAvg: 10 nodes, 4 rounds
  - scikit-learn with FedMedian: 6 nodes, 8 rounds, with injected failures
- The scene rendered in a real browser (Chromium, Claude desktop app's browser pane):
  - live during a run
  - replayed from JSONL with no federation running
- In both recordings:
  - every client message is matched to the server's send by id
  - node ids agree on both sides
  - every client event got its round
- Aggregated metrics in the events match Flower's own log exactly.
- Both recordings replay through the browser's reducer with all checks passing.
- 300 characters (synthetic): about 6.6 ms GPU render plus 3 ms update per frame on an Apple M3, 78 draw calls, about 550k triangles.

**Not verified**
- Real multi-machine deployment: SuperNodes on other hosts, `--host 0.0.0.0`, the token over a real network.
- Frame rate in a normal browser window. The embedded pane throttles frames: it showed 50–60 fps with 10 characters and 13–22 fps with 300, although the GPU timing above leaves room for 60.
- Flower versions other than 1.44.0. Firefox and Safari.
- `Bulyan`, `Krum`, the XGBoost strategies, and the DP strategy wrappers. The design covers them, but they have no test.

**Left to do**
- Publish to PyPI so apps don't need the `[tool.uv.sources]` path.
- A TLS option for the collector.
- Per-message particles past about 150 nodes are sampled, not aggregated into flow bands.

## The presentation site

`python tools/build_site.py` builds the static site into `site/`: the slides
from `site_src/`, plus the viewer in static mode with the recordings from
`examples/`. It is published on the `gh-pages` branch at https://qwertykeypadapple.github.io/flwr-viz/

## Licence

Apache-2.0. The two examples in `examples/` started from Flower's quickstart apps
(Apache-2.0, licence kept in each folder). The recordings in `examples/hub/` come
from runs of third-party Flower Hub apps whose code is not included here.
flwr-viz is an independent project; it is not made by or affiliated with Flower Labs.
