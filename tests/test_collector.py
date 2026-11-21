from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from flwr_viz.collector.server import Collector
from flwr_viz.schema import make_event
from flwr_viz.synth import generate


def srv(etype, rnd=None, mid=None, **data):
    return make_event(etype, role="server", node_id=0, run_id=5, round=rnd, msg_id=mid, data=data)


def cli(etype, mid, node=1, **data):
    return make_event(etype, role="client", node_id=node, run_id=5, msg_id=mid, data=data)


def test_round_fill_in_and_hold(tmp_path):
    c = Collector(tmp_path, hold_seconds=1.0)
    # client event arrives BEFORE the server's msg.sent: it is held, not dropped
    out = c.ingest([cli("client.recv", "m1")], now=100.0)
    assert out == [] and len(c.held) == 1
    out = c.ingest([srv("msg.sent", rnd=3, mid="m1", dst=1)], now=100.2)
    assert [e["type"] for e in out] == ["msg.sent", "client.recv"]
    assert out[1]["round"] == 3
    # already-known message id: filled immediately
    out = c.ingest([cli("client.end", "m1", ok=True)], now=100.3)
    assert out[0]["round"] == 3
    # never-matched event is released after the hold with round=None
    c.ingest([cli("client.recv", "orphan")], now=200.0)
    out = c.ingest([], now=201.5)
    assert out[0]["msg_id"] == "orphan" and out[0]["round"] is None
    seqs = [e["seq"] for e in c.history]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))


def test_invalid_events_rejected(tmp_path):
    c = Collector(tmp_path)
    bad = [{"v": 99}, {"v": 1, "type": "nope"}, "str",
           make_event("client.end", role="server", node_id=0, run_id=1)]  # wrong role
    assert c.ingest(bad) == [] and c.rejected == 4


def test_jsonl_written_per_run(tmp_path):
    c = Collector(tmp_path)
    evs = generate(clients=4, rounds=2, run_id=123)
    c.ingest(evs, now=1.0)
    c.ingest([], now=10.0)  # release anything still held
    files = list(tmp_path.glob("run-123-*.jsonl"))  # run ids are strings in v1
    assert len(files) == 1
    lines = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(lines) == len(evs)
    assert all("seq" in l and "rts" in l for l in lines)


async def _client(c: Collector) -> TestClient:
    client = TestClient(TestServer(c.make_app()))
    await client.start_server()
    return client


async def test_ingest_broadcasts_to_websocket(tmp_path):
    c = Collector(tmp_path)
    client = await _client(c)
    try:
        await client.post("/ingest", json=[srv("run.start", strategy="FedAvg")])
        ws = await client.ws_connect("/ws")
        hello = await ws.receive_json()
        assert hello["kind"] == "hello" and hello["mode"] == "live" and hello["schema"] == 1
        hist = await ws.receive_json()
        assert hist["kind"] == "history" and hist["events"][0]["type"] == "run.start"
        r = await client.post("/ingest", json=[srv("round.start", rnd=1, phase="train",
                                                   connected=[1], sampled=[1])])
        assert (await r.json())["accepted"] == 1
        live = await asyncio.wait_for(ws.receive_json(), 2)
        assert live["kind"] == "events" and live["events"][0]["type"] == "round.start"
        st = await (await client.get("/api/status")).json()
        assert st["events"] == 2
        runs = await (await client.get("/api/runs")).json()
        assert len(runs["runs"]) == 1
        body = await (await client.get(f"/api/runs/{runs['runs'][0]['name']}")).text()
        assert len(body.splitlines()) == 2
        assert (await client.get("/api/runs/..%2Fsecret.jsonl")).status == 404
        index = await client.get("/")
        assert index.status == 200 and "three" in (await index.text()).lower()
        await ws.close()
    finally:
        await client.close()


async def test_token_required(tmp_path):
    c = Collector(tmp_path, token="s3cret")
    client = await _client(c)
    try:
        r = await client.post("/ingest", json=[srv("run.start")])
        assert r.status == 401
        r = await client.post("/ingest", json=[srv("run.start")],
                              headers={"X-Flwr-Viz-Token": "s3cret"})
        assert r.status == 200
    finally:
        await client.close()


async def test_held_events_released_by_timer(tmp_path):
    c = Collector(tmp_path, hold_seconds=0.2)
    client = await _client(c)
    try:
        ws = await client.ws_connect("/ws")
        await ws.receive_json(); await ws.receive_json()
        await client.post("/ingest", json=[cli("client.recv", "never-sent")])
        msg = await asyncio.wait_for(ws.receive_json(), 3)
        assert msg["events"][0]["msg_id"] == "never-sent"
    finally:
        await client.close()


@pytest.mark.parametrize("mode", ["replay"])
async def test_replay_mode_serves_file_without_federation(tmp_path, mode):
    live = Collector(tmp_path)
    evs = generate(clients=6, rounds=3, run_id=77)
    live.ingest(evs, now=1.0); live.ingest([], now=10.0)
    for fh in live.files.values():
        fh.close()
    path = next(tmp_path.glob("run-77-*.jsonl"))
    recorded = [json.loads(l) for l in path.read_text().splitlines()]

    rep = Collector(tmp_path, mode=mode)
    assert rep.load_jsonl(path) == len(recorded)
    client = await _client(rep)
    try:
        ws = await client.ws_connect("/ws")
        hello = await ws.receive_json()
        assert hello["mode"] == "replay"
        got = []
        while True:
            m = await ws.receive_json()
            got += m["events"]
            if m.get("done"):
                break
        assert got == recorded  # byte-for-byte same events, same order
        r = await client.post("/ingest", json=[srv("run.start")])
        assert r.status == 409  # replay mode is read-only
    finally:
        await client.close()


def test_integer_ids_are_normalised_not_rejected(tmp_path):
    c = Collector(tmp_path)
    ev = make_event("msg.sent", role="server", node_id=0, run_id=1, round=1, msg_id="m", data={})
    ev["run_id"] = 810105674617115280
    ev["src"]["node_id"] = 0
    ev["data"] = {"dst": 2607856368701370143, "mtype": "train"}
    out = c.ingest([ev])
    assert c.rejected == 0 and out[0]["run_id"] == "810105674617115280"
    assert out[0]["data"]["dst"] == "2607856368701370143"
