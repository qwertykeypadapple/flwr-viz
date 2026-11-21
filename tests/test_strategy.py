from __future__ import annotations

import uuid
from typing import Iterable

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedMedian, FedProx, Strategy

from flwr_viz import instrument


def AR(d):
    return ArrayRecord({k: Array(np.asarray(v, dtype=np.float32)) for k, v in d.items()})


INIT = {"w": np.zeros((3, 2)), "b": np.zeros(3)}


class FakeRun:
    run_id = 99


class FakeGrid(Grid):
    """In-process grid. Healthy nodes add +node_offset to weights.

    ``silent`` nodes never answer (-> timeout). ``vanish`` nodes disappear from
    get_node_ids() right after a message is pushed to them (-> NODE_UNAVAILABLE).
    ``crash`` nodes reply with an error message.
    """

    def __init__(self, nodes, silent=(), vanish=(), crash=()):
        self.nodes = list(nodes)
        self.silent, self.vanish, self.crash = set(silent), set(vanish), set(crash)
        self.gone: set[int] = set()
        self.outbox: list[Message] = []

    def set_run(self, run):
        pass

    @property
    def run(self):
        return FakeRun()

    def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
        return Message(content=content, message_type=message_type, dst_node_id=dst_node_id,
                       group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        return [n for n in self.nodes if n not in self.gone]

    def push_messages(self, messages):
        ids = []
        for m in messages:
            mid = uuid.uuid4().hex
            m.metadata.__dict__["_message_id"] = mid
            ids.append(mid)
            dst = m.metadata.dst_node_id
            if dst in self.vanish:
                self.gone.add(dst)
            elif dst not in self.silent:
                self.outbox.append(m)
        return ids

    def pull_messages(self, reply_to_message_ids=None, *, message_ids=None):
        wanted = set(reply_to_message_ids or [])
        out, keep = [], []
        for m in self.outbox:
            if m.metadata.message_id in wanted:
                out.append(self._handle(m))
            else:
                keep.append(m)
        self.outbox = keep
        return out

    def send_and_receive(self, messages, *, timeout=None):
        ids = self.push_messages(list(messages))
        return self.pull_messages(ids)

    def _handle(self, m: Message) -> Message:
        from flwr.app import Error

        node = m.metadata.dst_node_id
        if node in self.crash:
            return Message(Error(code=2, reason="client crashed"), reply_to=m)
        if m.metadata.message_type.startswith("train"):
            arrs = {k: v.numpy() + (node % 10) for k, v in m.content["arrays"].items()}
            content = RecordDict({"arrays": AR(arrs),
                                  "metrics": MetricRecord({"train_loss": 1.0 / node,
                                                           "num-examples": 10})})
        else:
            content = RecordDict({"metrics": MetricRecord({"eval_acc": 0.5,
                                                           "num-examples": 10})})
        return Message(content, reply_to=m)


def _run(strategy, grid, rounds=2, timeout=1.2, evaluate_fn=None):
    return strategy.start(grid=grid, initial_arrays=AR(INIT), num_rounds=rounds,
                          timeout=timeout, train_config=ConfigRecord({"lr": 0.1}),
                          evaluate_fn=evaluate_fn)


def test_full_round_events_fedavg(capture):
    grid = FakeGrid([1, 2, 3])
    s = instrument(FedAvg(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=3))
    assert isinstance(s, FedAvg)  # same object, type unchanged
    calls = []
    res = _run(s, grid, evaluate_fn=lambda r, a: calls.append(r) or MetricRecord({"acc": 0.1 * r}))
    t = capture.types()
    assert t[0] == "run.start" and t[-1] == "run.end"
    rs = capture.of("run.start")[0]
    assert rs["data"]["strategy"] == "FedAvg" and rs["data"]["num_rounds"] == 2
    assert rs["run_id"] == "99" and len(rs["data"]["layers"]) == 2

    starts = capture.of("round.start")
    assert [(e["round"], e["data"]["phase"]) for e in starts] == [
        (1, "train"), (1, "evaluate"), (2, "train"), (2, "evaluate")]
    assert sorted(starts[0]["data"]["sampled"]) == ["1", "2", "3"]

    sent = capture.of("msg.sent")
    replies = capture.of("msg.reply")
    assert len(sent) == 12 and len(replies) == 12  # 3 nodes x 2 phases x 2 rounds
    assert {e["msg_id"] for e in sent} == {e["msg_id"] for e in replies}
    assert all(e["data"]["ok"] for e in replies)
    assert all(e["data"]["latency_s"] >= 0 for e in replies)

    # global model change per layer == ||new - old|| computed independently
    agg = [e for e in capture.of("agg.end") if e["data"]["phase"] == "train"]
    assert len(agg) == 2 and agg[0]["data"]["updated"] is True
    mean_shift = np.mean([1, 2, 3])  # FedAvg with equal num-examples
    w = {l["name"]: l for l in agg[0]["data"]["layers"]}
    assert w["w"]["delta_norm"] == pytest.approx(mean_shift * np.sqrt(6), rel=1e-5)
    assert agg[0]["data"]["metrics"]["train_loss"] == pytest.approx(np.mean([1, 1 / 2, 1 / 3]))

    ev = capture.of("eval.central")
    assert [e["round"] for e in ev] == [0, 1, 2] and calls == [0, 1, 2]
    assert [e["round"] for e in capture.of("round.end")] == [1, 2]
    assert capture.of("run.end")[0]["data"]["rounds_completed"] == 2
    assert res.arrays is not None


def test_timeouts_unavailable_and_crashes(capture):
    grid = FakeGrid([1, 2, 3, 4], silent=[2], vanish=[3], crash=[4])
    s = instrument(FedAvg(fraction_train=1.0, fraction_evaluate=0.0, min_available_nodes=4))
    _run(s, grid, rounds=1, timeout=1.2)
    timeouts = capture.of("msg.timeout")
    assert [e["data"]["dst"] for e in timeouts] == ["2"]
    bad = {e["data"]["from"]: e["data"] for e in capture.of("msg.reply") if not e["data"]["ok"]}
    assert bad["3"]["error_code"] == "NODE_UNAVAILABLE"
    assert bad["4"]["error_code"] == "CLIENT_APP_RAISED_EXCEPTION"
    a = capture.of("agg.start")[0]["data"]
    assert (a["n_ok"], a["n_failed"]) == (1, 3)


def test_run_error_is_emitted_and_reraised(capture):
    grid = FakeGrid([1, 2])

    def boom(r, a):
        if r == 1:
            raise ValueError("eval broke")
        return None

    s = instrument(FedAvg(min_available_nodes=2, fraction_evaluate=0.0))
    with pytest.raises(ValueError, match="eval broke"):
        _run(s, grid, evaluate_fn=boom)
    err = capture.of("run.error")[0]["data"]
    assert "eval broke" in err["error"] and capture.types()[-1] == "run.error"


@pytest.mark.parametrize("cls", [FedMedian, FedProx])
def test_other_builtin_strategies(capture, cls):
    kw = {"proximal_mu": 0.1} if cls is FedProx else {}
    s = instrument(cls(min_available_nodes=3, fraction_evaluate=0.0, **kw))
    _run(s, FakeGrid([11, 12, 13]), rounds=1)
    w = {l["name"]: l for l in
         [e for e in capture.of("agg.end") if e["data"]["phase"] == "train"][0]["data"]["layers"]}
    expected = 2 if cls is FedMedian else 2  # median and mean of shifts 1,2,3 are both 2
    assert w["w"]["delta_norm"] == pytest.approx(expected * np.sqrt(6), rel=1e-5)
    assert capture.of("run.start")[0]["data"]["strategy"] == cls.__name__


class RoundRobin(Strategy):
    """A custom strategy: one node per round, takes its weights as the new global model."""

    def __init__(self):
        self.extra = "kept"

    def configure_train(self, server_round, arrays, config, grid):
        nodes = sorted(grid.get_node_ids())
        node = nodes[(server_round - 1) % len(nodes)]
        return [Message(RecordDict({"arrays": arrays, "config": config}), message_type="train",
                        dst_node_id=node)]

    def aggregate_train(self, server_round, replies):
        ok = [r for r in replies if not r.has_error()]
        return (ok[0].content["arrays"], None) if ok else (None, None)

    def configure_evaluate(self, server_round, arrays, config, grid):
        return []

    def aggregate_evaluate(self, server_round, replies):
        return None

    def summary(self):
        pass

    def custom_method(self):
        return 42


def test_custom_strategy(capture):
    s = instrument(RoundRobin())
    assert s.custom_method() == 42 and s.extra == "kept"
    _run(s, FakeGrid([5, 6]), rounds=2)
    starts = [e for e in capture.of("round.start") if e["data"]["phase"] == "train"]
    assert [e["data"]["sampled"] for e in starts] == [["5"], ["6"]]
    agg = [e for e in capture.of("agg.end") if e["data"]["phase"] == "train"]
    assert agg[0]["data"]["layers"][0]["delta_norm"] == pytest.approx(5 * np.sqrt(6), rel=1e-5)


def test_instrument_twice_is_noop(capture):
    s = FedAvg(min_available_nodes=2, fraction_evaluate=0.0)
    assert instrument(instrument(s)) is s
    _run(s, FakeGrid([1, 2]), rounds=1)
    assert len(capture.of("run.start")) == 1


def test_level_off_leaves_strategy_untouched(capture):
    s = FedAvg()
    before = s.start
    instrument(s, level="off")
    assert s.start == before and not hasattr(s, "_flwr_viz_state")


def test_minimal_level_server(capture):
    s = instrument(FedAvg(min_available_nodes=2, fraction_evaluate=0.0), level="minimal")
    _run(s, FakeGrid([1, 2]), rounds=1)
    for e in capture.events:
        assert e["data"].get("layers", []) == [] and e["data"].get("metrics", {}) == {}


# ---------------------------------------------------------------- grid-only apps
def _query(grid, mtype, nodes):
    msgs = [Message(RecordDict({"arrays": AR(INIT)}), message_type=mtype, dst_node_id=n) for n in nodes]
    return list(grid.send_and_receive(msgs, timeout=1.2))


def test_instrument_grid_without_strategy(capture):
    """A ServerApp that drives the grid itself (like federated K-means)."""
    raw = FakeGrid([1, 2, 3], silent=[3])
    grid = instrument(raw)
    assert instrument(grid) is grid                      # idempotent
    assert list(grid.get_node_ids()) == [1, 2, 3]       # behaves like the grid
    _query(grid, "query.init", [1, 2])
    _query(grid, "query.stats", [1, 2])
    for _ in range(3):                                   # three clustering iterations
        _query(grid, "query.cluster", [1, 2, 3])
        _query(grid, "query.converged", [1, 2])
    grid._viz.finish()
    t = capture.types()
    assert t[0] == "run.start" and t[-1] == "run.end"
    assert capture.of("run.start")[0]["data"]["strategy"] == "ServerApp without a Strategy"
    starts = [(e["round"], e["data"]["phase"]) for e in capture.of("round.start")]
    assert starts == [(1, "query.init"), (1, "query.stats"), (1, "query.cluster"), (1, "query.converged"),
                      (2, "query.cluster"), (2, "query.converged"), (3, "query.cluster"), (3, "query.converged")]
    assert len(capture.of("msg.sent")) == 2 + 2 + 3 * 3 + 3 * 2
    assert [e["data"]["dst"] for e in capture.of("msg.timeout")] == ["3"] * 3   # silent node, every cluster step
    assert [e["round"] for e in capture.of("round.end")] == [1, 2, 3]
    sent_ids = {e["msg_id"] for e in capture.of("msg.sent")}
    assert all(e["msg_id"] in sent_ids for e in capture.of("msg.reply"))


def test_grid_and_strategy_together_no_duplicates(capture):
    """An app that sends its own messages, then also runs a Strategy on the same grid."""
    grid = instrument(FakeGrid([1, 2]))
    _query(grid, "query.init", [1, 2])
    s = instrument(FedAvg(min_available_nodes=2, fraction_evaluate=0.0))
    _run(s, grid, rounds=1)
    sent = capture.of("msg.sent")
    assert len(sent) == 2 + 2                            # 2 init + 2 train, nothing counted twice
    assert len({e["msg_id"] for e in sent}) == 4
    assert grid._viz.paused is False                     # grid recorder resumes after the strategy


def test_uint8_blob_is_not_treated_as_numbers(capture):
    from flwr_viz.stats import summarize
    blob = Array(np.frombuffer(b'{"learner": {"trees": [1, 2, 3]}}' * 20, dtype=np.uint8))
    st = summarize([("0", blob)], level="detailed")[0]
    assert st["params"] is None and "norm" not in st and st["bytes"] > 0
    # but an ordinary uint8 tensor with shape is still numbers
    img = Array(np.ones((4, 4), dtype=np.uint8))
    assert summarize([("q", img)])[0]["params"] == 16
