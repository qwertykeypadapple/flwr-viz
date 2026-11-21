from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flwr_viz import make_viz_mod, report_progress, viz_mod

def AR(d):
    return ArrayRecord({k: Array(v) for k, v in d.items()})


W_IN = {"fc.weight": np.ones((4, 3), dtype=np.float32), "fc.bias": np.zeros(4, dtype=np.float32)}


def _msg(run_id: int = 7) -> Message:
    content = RecordDict({
        "arrays": AR(W_IN),
        "config": ConfigRecord({"lr": 0.1, "secret-path": "/data/patients.csv"}),
    })
    return Message(content=content, dst_node_id=42, message_type="train")


def _ctx(**run_config) -> Context:
    return Context(run_id=7, node_id=42, node_config={"partition-id": 3},
                   state=RecordDict(), run_config=dict(run_config))


def _train(msg: Message, ctx: Context) -> Message:
    arrs = msg.content["arrays"].to_numpy_ndarrays()
    new = [a + 0.5 for a in arrs]  # every element moves by 0.5
    names = list(msg.content["arrays"].keys())
    for e in range(2):
        for b in range(3):
            report_progress(epoch=e, batch=b, total_batches=3, loss=1.0 / (1 + b))
    return Message(RecordDict({
        "arrays": AR(dict(zip(names, new))),
        "metrics": MetricRecord({"train_loss": 0.25, "num-examples": 120}),
    }), reply_to=msg)


def test_event_sequence_and_layer_stats(capture):
    reply = viz_mod(_msg(), _ctx(), _train)
    assert not reply.has_error()
    types = capture.types()
    assert types[0] == "client.recv" and types[1] == "client.start" and types[-1] == "client.end"
    assert "client.progress" in types
    recv = capture.of("client.recv")[0]
    assert recv["src"] == {"role": "client", "node_id": "42"}
    assert recv["run_id"] == "7" and recv["data"]["mtype"] == "train"
    lw = {l["name"]: l for l in recv["data"]["layers"]}
    assert lw["fc.weight"]["shape"] == [4, 3] and lw["fc.weight"]["params"] == 12
    assert lw["fc.weight"]["norm"] == pytest.approx(np.sqrt(12))

    end = capture.of("client.end")[0]["data"]
    assert end["ok"] is True and end["duration_s"] >= 0
    assert end["metrics"] == {"train_loss": 0.25, "num-examples": 120}
    out = {l["name"]: l for l in end["layers"]}
    # delta = 0.5 everywhere -> ||delta|| = 0.5 * sqrt(n)
    assert out["fc.weight"]["delta_norm"] == pytest.approx(0.5 * np.sqrt(12))
    assert out["fc.bias"]["delta_norm"] == pytest.approx(0.5 * 2)
    assert out["fc.bias"]["norm"] == pytest.approx(0.5 * 2)


def test_never_emits_raw_values_or_config(capture):
    big = {"w": np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)}
    msg = Message(RecordDict({"arrays": AR(big),
                              "config": ConfigRecord({"secret-path": "/data/patients.csv"})}),
                  dst_node_id=42, message_type="train")
    viz_mod(msg, _ctx(**{"viz-level": "detailed"}), lambda m, c: Message(
        RecordDict({"arrays": AR(big)}), reply_to=m))
    blob = json.dumps(capture.events)
    assert "patients" not in blob and "partition" not in blob
    # 4096 floats would be far more than this; we only ship ~10 numbers per layer
    assert len(blob) < 3000
    st = capture.of("client.end")[0]["data"]["layers"][0]
    assert set(st) <= {"name", "dtype", "bytes", "shape", "params", "norm", "delta_norm",
                       "mean", "std", "min", "max"}
    assert st["std"] == pytest.approx(float(big["w"].astype(np.float64).std()), rel=1e-6)


def test_minimal_level_has_no_metrics_or_layers(capture):
    viz_mod(_msg(), _ctx(**{"viz-level": "minimal"}), _train)
    for e in capture.events:
        assert e["data"].get("layers", []) == []
        assert e["data"].get("metrics", {}) == {}
        assert "loss" not in e["data"]


def test_off_level_emits_nothing(capture):
    reply = viz_mod(_msg(), _ctx(**{"viz-level": "off"}), _train)
    assert not reply.has_error() and capture.events == []


def test_layer_names_can_be_hidden(capture):
    viz_mod(_msg(), _ctx(**{"viz-layer-names": False}), _train)
    names = [l["name"] for l in capture.of("client.end")[0]["data"]["layers"]]
    assert names == ["layer_0", "layer_1"]


def test_exception_passes_through_unchanged(capture):
    class Boom(RuntimeError):
        pass

    def bad(msg, ctx):
        raise Boom("disk full\nsecond line not sent")

    with pytest.raises(Boom):
        viz_mod(_msg(), _ctx(), bad)
    end = capture.of("client.end")[0]["data"]
    assert end["ok"] is False and end["error"] == "Boom: disk full"


def test_error_reply_is_reported(capture):
    from flwr.app import Error

    viz_mod(_msg(), _ctx(), lambda m, c: Message(Error(code=2, reason="oops"), reply_to=m))
    end = capture.of("client.end")[0]["data"]
    assert end["ok"] is False and "oops" in end["error"]


def test_progress_outside_mod_is_noop(capture):
    report_progress(epoch=0, batch=0, loss=1.0)
    assert capture.events == []


def test_progress_rate_limit_and_boundaries(capture):
    def loop(msg, ctx):
        for b in range(1000):
            report_progress(epoch=0, batch=b, total_batches=1000, loss=1.0)
        return Message(RecordDict(), reply_to=msg)

    viz_mod(_msg(), _ctx(**{"viz-progress-hz": 5}), loop)
    prog = capture.of("client.progress")
    batches = [p["data"]["batch"] for p in prog]
    assert batches[0] == 0 and batches[-1] == 999  # first and last always sent
    assert len(prog) < 10  # the loop takes a few ms, so ~2-3 events


def test_progress_from_worker_thread(capture):
    def loop(msg, ctx):
        t = threading.Thread(target=lambda: report_progress(epoch=0, batch=0, loss=0.5))
        t.start(); t.join()
        return Message(RecordDict(), reply_to=msg)

    viz_mod(_msg(), _ctx(), loop)
    assert len(capture.of("client.progress")) == 1
    assert capture.of("client.progress")[0]["src"]["node_id"] == "42"


def test_non_numeric_arrays_get_bytes_only(capture):
    blob = Array(dtype="bytes", shape=(1,), stype="raw", data=b"xgboost-model-bytes" * 10)
    msg = Message(RecordDict({"arrays": ArrayRecord({"model": blob})}), dst_node_id=1,
                  message_type="train")
    viz_mod(msg, _ctx(), lambda m, c: Message(RecordDict({"arrays": ArrayRecord({"model": blob})}),
                                              reply_to=m))
    st = capture.of("client.end")[0]["data"]["layers"][0]
    assert st["bytes"] == 190 and st["params"] is None and "norm" not in st


def test_mod_works_inside_a_real_clientapp(capture):
    app = ClientApp(mods=[make_viz_mod()])

    @app.train()
    def train(msg, ctx):
        return _train(msg, ctx)

    reply = app(_msg(), _ctx())
    assert not reply.has_error()
    assert capture.types()[0] == "client.recv" and capture.types()[-1] == "client.end"


def test_64bit_ids_are_strings(capture):
    big = 2607856368701370143  # a real Flower node id, > 2**53
    ctx = Context(run_id=810105674617115280, node_id=big, node_config={}, state=RecordDict(),
                  run_config={})
    viz_mod(_msg(), ctx, _train)
    e = capture.events[0]
    assert e["src"]["node_id"] == str(big) and e["run_id"] == "810105674617115280"


def test_delta_matches_layers_across_different_record_counts(capture):
    """Message carries two ArrayRecords (model + extra), reply carries one (FedMAP does this)."""
    msg = Message(RecordDict({"arrays": AR(W_IN), "extra": AR({"prior": np.zeros(3, dtype=np.float32)})}),
                  dst_node_id=42, message_type="train")
    viz_mod(msg, _ctx(), lambda m, c: Message(RecordDict({
        "arrays": AR({k: v + 0.5 for k, v in W_IN.items()})}), reply_to=m))
    out = {l["name"]: l for l in capture.of("client.end")[0]["data"]["layers"]}
    assert out["fc.weight"]["delta_norm"] == pytest.approx(0.5 * np.sqrt(12))
    recv = [l["name"] for l in capture.of("client.recv")[0]["data"]["layers"]]
    assert recv == ["arrays/fc.weight", "arrays/fc.bias", "extra/prior"]
