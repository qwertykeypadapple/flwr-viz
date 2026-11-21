# flwr-viz event schema, version 1

Every part of flwr-viz speaks this one format: the client mod and the strategy
wrapper produce it, the collector validates, records and broadcasts it, and the
browser rebuilds its whole view from it. A recorded run is a JSONL file of these
events, one per line.

Source of truth in code: `src/flwr_viz/schema.py` (`SCHEMA_VERSION`, `EVENT_TYPES`,
`validate_event`).

## Envelope

```json
{
  "v": 1,
  "type": "client.end",
  "ts": 1795262067.873,
  "run_id": "810105674617115280",
  "src": {"role": "client", "node_id": "2607856368701370143"},
  "round": 3,
  "msg_id": "8f1c…",
  "data": { }
}
```

| field | type | meaning |
|---|---|---|
| `v` | int | schema version, `1` |
| `type` | string | one of the event types below |
| `ts` | float | sender's wall clock, Unix seconds |
| `run_id` | string or null | Flower run id |
| `src.role` | `"server"` or `"client"` | who emitted it |
| `src.node_id` | string or null | Flower node id of the sender; the server uses `"0"` |
| `round` | int or null | server round. Clients usually don't know it; the collector fills it in (see below) |
| `msg_id` | string or null | Flower message id this event is about |
| `data` | object | per-type payload |

**Ids are decimal strings.** Flower run and node ids are 64-bit integers.
JavaScript parses JSON numbers as doubles, which are exact only up to 2^53, so
`2607856368701370143` would turn into `2607856368701370000`. Every id field
(`run_id`, `src.node_id`, `data.dst`, `data.from`, `data.sampled[]`,
`data.connected[]`) is therefore a string. The collector converts integer ids it
receives from older or hand-written emitters into strings rather than rejecting them.

### Fields added by the collector

| field | meaning |
|---|---|
| `seq` | int, total order of arrival at the collector, starting at 1. Use it to order events. |
| `rts` | float, collector receive time |
| `round` | filled in on client events from the matching `msg.sent` (same `msg_id`) |

Client events that arrive before their `msg.sent` are held for up to 1.5 s
waiting for it, then released with `round: null` if it never comes.

## Event types

### Server (strategy wrapper)

| type | when | `data` |
|---|---|---|
| `run.start` | `strategy.start()` called | `strategy` (class name), `num_rounds`, `timeout`, `level`, `layers: [LayerStat]` (initial model), `connected: [id]` |
| `round.start` | `configure_train` / `configure_evaluate` returned | `phase` (`"train"` or `"evaluate"`), `connected: [id]`, `sampled: [id]` |
| `msg.sent` | message pushed to the SuperLink | `dst`, `mtype` (Flower message type), `phase`, `bytes`, `pushed` (false if Flower returned no id) |
| `msg.reply` | reply pulled, or error reply seen at aggregation | `from`, `ok`, `latency_s` (server clock, push→pull), `phase`, `bytes`; when not ok: `error_code` (e.g. `NODE_UNAVAILABLE`, `CLIENT_APP_RAISED_EXCEPTION`), `error_reason` (≤200 chars) |
| `msg.timeout` | aggregation started and this message never got a reply | `dst`, `waited_s`, `phase` |
| `agg.start` | `aggregate_*` called | `phase`, `n_replies`, `n_ok`, `n_failed` (failed includes timeouts) |
| `agg.end` | `aggregate_*` returned | `phase`, `duration_s`, `metrics{}`; train only: `updated` (bool), `layers: [LayerStat]` where `delta_norm` = ‖new global − previous global‖ per layer |
| `eval.central` | user's `evaluate_fn` returned (round 0 = initial model) | `metrics{}`, `duration_s` |
| `round.end` | next round starts, or the run ends | `duration_s` |
| `run.end` | `start()` returned | `duration_s`, `rounds_completed` |
| `run.error` | `start()` raised (the exception is re-raised) | `duration_s`, `rounds_completed`, `error` (type + first 200 chars) |

### Client (mod)

| type | when | `data` |
|---|---|---|
| `client.recv` | message enters the mod | `mtype`, `bytes`, `layers: [LayerStat]` (as received) |
| `client.start` | right before the handler runs | `mtype` |
| `client.progress` | user called `report_progress()` (rate-limited) | `epoch`, `batch`, `total_batches`, `loss`, `extra{}` (numbers only; `loss`/`extra` omitted at level `minimal`) |
| `client.end` | handler returned or raised | `mtype`, `ok`, `duration_s` (client clock), `metrics{}`, `bytes`, `layers: [LayerStat]` with `delta_norm` = ‖sent − received‖ per layer; when not ok: `error` |

### Any

| type | `data` |
|---|---|
| `sys.dropped` | `count`: events this process dropped (collector down, queue full) since the last report |

## LayerStat

One entry per array in the message's `ArrayRecord`s, in order. Names are the
array keys, or `record/key` when a message carries more than one ArrayRecord.

| field | present | meaning |
|---|---|---|
| `name` | always | array name, or `layer_<i>` when `viz-layer-names = false` |
| `dtype` | always | Flower `Array.dtype` |
| `bytes` | always | serialized size |
| `params` | always | element count; `null` for non-numeric arrays (e.g. XGBoost model bytes) |
| `shape` | numeric arrays | list of ints |
| `norm` | numeric arrays | L2 norm |
| `delta_norm` | when a reference exists with the same name and shape | L2 norm of the difference |
| `mean`, `std`, `min`, `max` | level `detailed` only | |
| `merged` | the catch-all entry | set when the model has more than `viz-max-layers` arrays (default 64): the rest are combined into one entry named `(other N arrays)` with summed params/bytes and combined norms |

## Detail levels

Set with `viz-level` in the run config, `FLWR_VIZ_LEVEL`, or the `level=` argument.

| level | what leaves the process |
|---|---|
| `off` | nothing; the mod and wrapper do no work |
| `minimal` | timing, message flow, node states, errors, byte counts. No metrics, no layer stats, no loss |
| `standard` (default) | adds metrics and per-layer name/shape/params/norm/delta_norm |
| `detailed` | adds per-layer mean/std/min/max |

Never sent at any level: raw weights or gradients, training data or labels,
`ConfigRecord` contents, `node_config` (partition ids etc.), tracebacks.

## Compatibility

Consumers must ignore unknown fields in `data`. New optional `data` fields can be
added within version 1. Removing or renaming fields, or changing a type, needs
`v: 2`. The collector rejects events with an unknown `v` or `type`.
