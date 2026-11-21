"""The collector being down, unroutable, or slow must never hurt training."""

from __future__ import annotations

import http.server
import socket
import threading
import time

import numpy as np
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict

from flwr_viz.emitter import Emitter
from flwr_viz.mod import make_viz_mod
from flwr_viz.schema import make_event


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port  # nothing listens here afterwards -> connection refused


def _ev(i: int):
    return make_event("client.progress", role="client", node_id=1, run_id=1,
                      data={"epoch": 0, "batch": i})


def test_emit_is_fast_when_collector_refuses_connections():
    em = Emitter(f"http://127.0.0.1:{_free_port()}")
    t0 = time.perf_counter()
    for i in range(20_000):
        em.emit(_ev(i))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, elapsed  # ~ microseconds per event, no network wait
    time.sleep(0.5)
    assert em.failures >= 1 and em.dropped > 0 and em.sent == 0


def test_unroutable_address_does_not_block_and_exit_flush_is_bounded():
    em = Emitter("http://10.255.255.1:7878")  # blackhole: connect hangs until timeout
    t0 = time.perf_counter()
    for i in range(1000):
        em.emit(_ev(i))
    assert time.perf_counter() - t0 < 0.5
    t1 = time.perf_counter()
    em.flush(1.0)
    assert time.perf_counter() - t1 < 1.8  # capped, even though nothing can be sent


def test_queue_overflow_drops_instead_of_blocking():
    em = Emitter(f"http://127.0.0.1:{_free_port()}", maxsize=10)
    em._ensure_thread = lambda: None  # no sender thread: queue can only fill up
    for i in range(1000):
        em.emit(_ev(i))
    assert em.dropped == 990


class _Slow(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        time.sleep(3)
        self.send_response(200); self.end_headers()

    def log_message(self, *a):
        pass


def test_slow_collector_does_not_slow_emit():
    port = _free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        em = Emitter(f"http://127.0.0.1:{port}")
        t0 = time.perf_counter()
        for i in range(5000):
            em.emit(_ev(i))
        assert time.perf_counter() - t0 < 0.5
    finally:
        srv.shutdown()


def test_recovers_and_reports_drops_when_collector_comes_back(monkeypatch):
    import flwr_viz.emitter as emod

    monkeypatch.setattr(emod, "_BACKOFF_STEPS", (0.2,))
    got: list = []

    class _Ok(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            import json
            n = int(self.headers["Content-Length"])
            got.extend(json.loads(self.rfile.read(n)))
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    port = _free_port()
    em = Emitter(f"http://127.0.0.1:{port}")
    for i in range(50):
        em.emit(_ev(i))
    time.sleep(0.4)  # these fail: nothing listening
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Ok)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        time.sleep(0.3)
        em.emit(_ev(999))
        deadline = time.time() + 3
        while time.time() < deadline and not any(e["type"] == "sys.dropped" for e in got):
            time.sleep(0.05)
        dropped = [e for e in got if e["type"] == "sys.dropped"]
        assert dropped and dropped[0]["data"]["count"] >= 50
        assert any(e["data"].get("batch") == 999 for e in got)
    finally:
        srv.shutdown()


def test_training_overhead_with_dead_collector():
    """A full mod round trip on a ~1M-parameter model against a dead collector."""
    from flwr.supercore.task_identity import TaskIdentity

    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 1, 0, 1
    rng = np.random.default_rng(0)
    W = {f"l{i}": rng.normal(size=(500, 500)).astype(np.float32) for i in range(4)}
    msg = Message(RecordDict({"arrays": ArrayRecord({k: Array(v) for k, v in W.items()})}),
                  dst_node_id=1, message_type="train")
    ctx = Context(run_id=1, node_id=1, node_config={}, state=RecordDict(),
                  run_config={"viz-url": f"http://127.0.0.1:{_free_port()}"})

    def handler(m, c):
        new = {k: Array(v + 0.01) for k, v in W.items()}
        return Message(RecordDict({"arrays": ArrayRecord(new),
                                   "metrics": MetricRecord({"loss": 1.0})}), reply_to=m)

    mod = make_viz_mod()
    t0 = time.perf_counter(); handler(msg, ctx); base = time.perf_counter() - t0
    times = []
    for _ in range(5):
        t0 = time.perf_counter(); mod(msg, ctx, handler); times.append(time.perf_counter() - t0)
    overhead = min(times) - base
    print(f"\nhandler {base*1e3:.1f} ms, with viz_mod {min(times)*1e3:.1f} ms "
          f"(overhead {overhead*1e3:.1f} ms for 1M params)")
    assert overhead < 0.25  # stats cost for 1M params, network adds nothing
