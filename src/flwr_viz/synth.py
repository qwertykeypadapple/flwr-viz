"""Synthetic event generator, for testing the UI at scale without a federation.

The events follow the real schema but every number is made up. It is a load
and rendering test only; it proves nothing about Flower.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
import uuid
from typing import Any

from .schema import make_event

LAYERS = [("conv1.weight", [6, 3, 5, 5]), ("conv1.bias", [6]), ("conv2.weight", [16, 6, 5, 5]),
          ("conv2.bias", [16]), ("fc1.weight", [120, 400]), ("fc1.bias", [120]),
          ("fc2.weight", [84, 120]), ("fc2.bias", [84]), ("fc3.weight", [10, 84]),
          ("fc3.bias", [10])]


def _layers(scale: float, delta: float | None) -> list[dict[str, Any]]:
    out = []
    for name, shape in LAYERS:
        n = 1
        for s in shape:
            n *= s
        norm = scale * (n ** 0.5) * 0.05
        st = {"name": name, "shape": shape, "dtype": "float32", "params": n, "bytes": n * 4,
              "norm": norm}
        if delta is not None:
            st["delta_norm"] = norm * delta * random.uniform(0.3, 1.7)
        out.append(st)
    return out


def generate(clients: int = 50, rounds: int = 5, fraction: float = 0.5,
             fail_rate: float = 0.05, run_id: int | None = None) -> list[dict[str, Any]]:
    """Return a list of (ts-ordered) events for a fake run starting now."""
    rid = str(run_id or random.randint(10**17, 2**63 - 1))
    nodes = [str(random.randint(10**17, 2**63 - 1)) for _ in range(clients)]  # 64-bit like Flower
    t = time.time()
    evs: list[dict[str, Any]] = []

    def srv(etype: str, data: dict[str, Any], rnd: int | None, ts: float, mid: str | None = None):
        evs.append(make_event(etype, role="server", node_id=0, run_id=rid, round=rnd,
                              msg_id=mid, data=data, ts=ts))

    def cli(node: int, etype: str, data: dict[str, Any], ts: float, mid: str):
        evs.append(make_event(etype, role="client", node_id=node, run_id=rid, msg_id=mid,
                              data=data, ts=ts))

    srv("run.start", {"strategy": "SyntheticFedAvg", "num_rounds": rounds, "timeout": 60,
                      "level": "standard", "layers": _layers(1.0, None), "connected": nodes},
        None, t)
    loss = 2.3
    for r in range(1, rounds + 1):
        t += 0.3
        for phase in ("train", "evaluate"):
            k = max(1, int(clients * fraction))
            sampled = random.sample(nodes, k)
            srv("round.start", {"phase": phase, "connected": nodes, "sampled": sampled}, r, t)
            ends = []
            for n in sampled:
                mid = uuid.uuid4().hex
                ts = t + random.uniform(0, 0.2)
                srv("msg.sent", {"dst": n, "mtype": phase, "phase": phase, "bytes": 250_000,
                                 "pushed": True}, r, ts, mid)
                if random.random() < fail_rate / 2:
                    continue  # timeout: never replies
                tr = ts + random.uniform(0.05, 0.3)
                cli(n, "client.recv", {"mtype": phase, "bytes": 250_000,
                                       "layers": _layers(1.0 + 0.1 * r, None)}, tr, mid)
                cli(n, "client.start", {"mtype": phase}, tr + 0.01, mid)
                dur = random.uniform(1.0, 3.0) if phase == "train" else random.uniform(0.3, 0.8)
                if phase == "train":
                    for e in range(2):
                        for b in range(0, 50, 10):
                            cli(n, "client.progress", {
                                "epoch": e, "batch": b, "total_batches": 50,
                                "loss": loss * random.uniform(0.8, 1.1) * (1 - 0.1 * e - b / 600),
                                "extra": {}}, tr + dur * (e * 50 + b) / 100, mid)
                failed = random.random() < fail_rate / 2
                te = tr + dur
                metrics = ({"train_loss": loss * random.uniform(0.85, 1.1), "num-examples": 600}
                           if phase == "train" else
                           {"eval_loss": loss * random.uniform(0.9, 1.1),
                            "eval_acc": min(0.99, 0.1 + 0.15 * r * random.uniform(0.8, 1.1)),
                            "num-examples": 150})
                cli(n, "client.end", {
                    "mtype": phase, "ok": not failed, "duration_s": dur,
                    "error": "RuntimeError: synthetic failure" if failed else None,
                    "metrics": {} if failed else metrics, "bytes": 0 if failed else 250_000,
                    "layers": [] if failed or phase != "train" else _layers(1.0 + 0.1 * r, 0.2 / r),
                }, te, mid)
                srv("msg.reply", {"from": n, "ok": not failed, "latency_s": te - ts + 0.05,
                                  "phase": phase, "bytes": 0 if failed else 250_000,
                                  **({"error_code": "CLIENT_APP_RAISED_EXCEPTION",
                                      "error_reason": "synthetic"} if failed else {})},
                    r, te + 0.05, mid)
                ends.append(te + 0.05)
            t = max(ends) if ends else t + 1
            srv("agg.start", {"phase": phase, "n_replies": len(ends), "n_ok": len(ends),
                              "n_failed": k - len(ends)}, r, t)
            t += 0.2
            data = {"phase": phase, "duration_s": 0.2,
                    "metrics": {"train_loss": loss} if phase == "train" else
                    {"eval_acc": 0.1 + 0.15 * r, "eval_loss": loss}}
            if phase == "train":
                data["layers"] = _layers(1.0 + 0.1 * r, 0.15 / r)
                data["updated"] = True
            srv("agg.end", data, r, t)
        srv("eval.central", {"metrics": {"accuracy": min(0.99, 0.12 + 0.14 * r), "loss": loss},
                             "duration_s": 0.4}, r, t + 0.4)
        t += 0.5
        srv("round.end", {"duration_s": 5.0}, r, t)
        loss *= 0.75
    srv("run.end", {"duration_s": t - evs[0]["ts"], "rounds_completed": rounds}, None, t + 0.1)
    evs.sort(key=lambda e: e["ts"])
    return evs


def stream(url: str, speed: float = 1.0, **kwargs: Any) -> int:
    """Generate a run and POST it to a collector in (scaled) real time."""
    evs = generate(**kwargs)
    t0, w0 = evs[0]["ts"], time.time()
    batch: list[dict[str, Any]] = []
    for ev in evs:
        due = w0 + (ev["ts"] - t0) / speed
        ev["ts"] = due
        if batch and time.time() < due:
            _post(url, batch)
            batch = []
            time.sleep(max(0.0, due - time.time()))
        batch.append(ev)
    if batch:
        _post(url, batch)
    return len(evs)


def _post(url: str, batch: list[dict[str, Any]]) -> None:
    req = urllib.request.Request(url.rstrip("/") + "/ingest",
                                 data=json.dumps(batch).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()
