# flwr-viz on Flower Hub apps (2026-11-21, Flower 1.44.0, simulation)

Four apps were downloaded with `flwr new` from Flower Hub (238 apps listed by
`api.flower.ai/v1/hub/apps`), checked for risky code (no network calls,
subprocesses or `eval`/`exec` beyond PyTorch's `model.eval()`), enabled with the
fewest changes possible, and run in simulation with the collector live.
Untouched copies are in `_template_ref/hub/` so every change can be diffed.
Recordings are in `examples/hub/<app>/recorded-run*.jsonl`. Check them with
`python tools/inspect_run.py <file>` and `node tests/js/replay_check.mjs <file>`.

| app | kind | strategy | nodes / rounds | result |
|---|---|---|---|---|
| `@flwrlabs/federated-vae` | PyTorch VAE on CIFAR-10 | FedAvg | 5 / 3 | ran clean, 173 events, replays |
| `@swiergarst/fed-kmeans-flower` | federated K-means (NumPy) | **none**: own loop with 7 `query.*` message types | 5 / converged in 2 | ran clean after 2 plug-in fixes, 160 events, replays |
| `@bloodcounts/nhanes-t2d-fedmap-fl` | PyTorch MLP, NHANES T2D | **FedMAP**, a custom `Strategy` subclass sending 2 ArrayRecords | 10 / 4 (app default 100) | ran clean, 435 events, replays; exposed a layer-matching bug (fixed, re-run) |
| `@flwrlabs/quickstart-xgboost` | XGBoost on Higgs | FedXgbBagging | 5 / 3 | **could not train: disk full** while unpacking Higgs (10.5M rows); the failing run was recorded |

## Changes each app needed

All apps got the same `pyproject.toml` addition: `"flwr-viz"` in dependencies,
`viz-url` in the run config, and a `[tool.uv.sources]` path. The path is needed
only until flwr-viz is on PyPI.

- **federated-vae:** 2 code lines plus their 2 imports. `ClientApp(mods=[viz_mod])` and
  `strategy = instrument(FedAvg(), run_config=context.run_config)`.
- **nhanes-t2d-fedmap-fl:** 2 code lines plus 2 imports, the same two, wrapping `FedMAP(...)`.
  Nothing in the custom strategy changed.
- **quickstart-xgboost:** 2 code lines plus 2 imports, the same two, wrapping `FedXgbBagging(...)`.
- **fed-kmeans-flower:** 2 code lines plus 2 imports. `ClientApp(mods=[viz_mod])` and, as the
  first line of `main`, `grid = instrument(grid, run_config=context.run_config)`.
  This needed a new plug-in feature (below). Not plug-in related: the app's
  `data_path` pointed at the author's home folder, so I generated data with its
  own `generate_test_data.py` and passed `--run-config data_path=...` (no file edit).

## What the scene showed

- **VAE:**
  - 14 named layers per update.
  - Pale-blue to gold books, because a VAE moves most layers 0.01–2.6% per
    round, with a few decoder biases up to 18%.
  - Lavender evaluating phase, loss in the server charts.
  - Train metrics are empty because the app returns only `num-examples` for
    training. That is correct.
- **K-means:**
  - The feed narrates the app's own steps: `initialize_client` →
    `compute_local_statistics` → `perform_local_clustering` →
    `evaluate_convergence`, repeated.
  - Rounds were inferred (2) and match the app's own log ("Round 1/50", "Round 2/50").
  - Client cards chart each job's results (ARI, silhouette, inertia, cluster sizes).
  - No aggregation animation and no global-model books: the app merges cluster
    means in its own code, which the plug-in cannot see. The feed and the
    explanations say so.
- **FedMAP:**
  - 10 clients, 4 training at a time.
  - Central evaluation per round (AUROC 0.55 → 0.79).
  - The received model shows 64 entries: the model's 10 layers plus the
    per-layer prior networks FedMAP sends in a second record, capped at 64.
  - Client updates show 10 layers with 9–98% change (red books).
  - Global changes are 22–100% per round.
- **XGBoost:**
  - All 30 jobs failed: every client and every round.
  - The scene showed red, dizzy clients, error replies, empty aggregations, and
    the run ending normally.

## What broke or looked wrong, and the general fix

Each fix is in the plug-in, not in any app, and has a test.

1. **Apps without a Strategy were invisible on the server side** (K-means).
   `instrument()` now also accepts a `Grid`. It records every message and
   reply, infers timeouts and rounds (a new round starts when a message type
   repeats), and emits run start/end. If a Strategy later runs on the same grid,
   its recorder takes over, with no double counting.
   (`tests/test_strategy.py::test_instrument_grid_*`)
2. **flwr-viz pinned `flwr>=1.37`, which broke `uv sync`** for an app that
   supports Python 3.10 (K-means; Flower ≥1.37 needs 3.11). flwr-viz now depends
   on plain `flwr` and warns at import if the version is older than 1.37.
3. **Per-layer change was missing when message and reply carry different numbers
   of ArrayRecords** (FedMAP: model + priors in, model out, so names were
   `arrays/net.0.weight` against `net.0.weight`). Layers are now matched with or
   without the record prefix when the bare name is unique.
   (`tests/test_mod.py::test_delta_matches_layers_across_different_record_counts`)
4. **Errors hid their cause** (XGBoost). Clients showed only "DatasetGenerationError:
   An error occurred while generating the dataset", and the server got 200
   characters of Ray wrapper text. Both now report the root cause:
   `… (caused by OSError: [Errno 28] No space left on device)`.
   (`tests/test_errors.py`)
5. **Serialized byte payloads were treated as numbers** (XGBoost ships its model
   as a 1-D `uint8` array, so a "norm" of raw bytes would be shown). Flat `uint8`
   arrays are now reported by size only. Shaped `uint8` tensors still count as
   numbers. (`tests/test_strategy.py::test_uint8_blob_is_not_treated_as_numbers`)
6. **Custom message types all collapsed into one phase "query"** (K-means), so
   one job's metrics overwrote another's, custom jobs were drawn as "training",
   the server card only had train/evaluate columns, and the timeline had no
   round markers. These are now handled generally:
   - Custom jobs keep their own names.
   - There is a new "working" state (teal shirt, busy at a keyboard).
   - The server card shows one column per job kind.
   - Client cards chart every job kind.
   - Round markers are placed at the first step of each round, whatever its phase.
7. Small: the feed said "? rounds" for apps that don't declare a round count, and
   the hover explanation said a large change is "normal in round 1" even in
   later rounds. The first is fixed; the second is still generic text.

## Not verified

- **XGBoost training.** The machine's disk was full (1.3–1.9 GB free) and Higgs
  needs several GB to unpack. So the byte-payload fix is covered by a unit test
  and by the recorded failing run's initial empty model, not by a successful
  XGBoost round. This session's Higgs download (2.0 GB) was deleted afterwards.
- Only 4 of the 154 server apps were tried. No LLM fine-tuning (FlowerTune)
  apps and no AgentApps.
