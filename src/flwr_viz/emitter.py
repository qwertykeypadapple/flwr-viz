"""Non-blocking event sender.

Contract: ``emit()`` never blocks on the network and never raises. If the
collector is slow or down, events are dropped and counted; training goes on.
Only the standard library is used, so nothing extra is imported into the
training process.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import urllib.request
from typing import Any

from .schema import make_event

_BATCH_MAX = 200
_BATCH_INTERVAL = 0.1
_HTTP_TIMEOUT = 0.5
_BACKOFF_STEPS = (2.0, 5.0, 10.0, 30.0)
_EXIT_FLUSH_CAP = 1.0


def _json_default(o: Any) -> Any:
    # numpy scalars / arrays of stats, sets, anything odd -> best effort
    try:
        import numpy as np  # local import, numpy is already a dependency

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class Emitter:
    def __init__(self, url: str, token: str | None = None, maxsize: int = 10_000) -> None:
        self.url = url.rstrip("/") + "/ingest"
        self.token = token
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pid = -1
        self._stop = threading.Event()
        self._next_try = 0.0
        self._fail_streak = 0
        self._last_src: tuple[str, int | None, int | None] = ("client", None, None)
        # counters (read by tests and by sys.dropped)
        self.sent = 0
        self.dropped = 0
        self.failures = 0
        self._dropped_unreported = 0

    # -- public -----------------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._ensure_thread()
            src = event.get("src") or {}
            self._last_src = (src.get("role", "client"), src.get("node_id"), event.get("run_id"))
            self._q.put_nowait(event)
        except queue.Full:
            self._count_drop(1)
        except Exception:  # never let telemetry hurt training
            self._count_drop(1)

    def flush(self, timeout: float = _EXIT_FLUSH_CAP) -> None:
        """Best-effort: send what is queued, giving up after ``timeout``."""
        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            if time.monotonic() < self._next_try:
                break  # collector known to be down; don't wait for it
            self._send_once(max_wait=0.0)

    def close(self) -> None:
        self.flush()
        self._stop.set()

    # -- internals --------------------------------------------------------
    def _count_drop(self, n: int) -> None:
        self.dropped += n
        self._dropped_unreported += n

    def _ensure_thread(self) -> None:
        pid = os.getpid()
        if self._thread is not None and self._thread.is_alive() and self._pid == pid:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._pid == pid:
                return
            if self._pid != pid and self._pid != -1:
                # forked child: start with a fresh queue
                self._q = queue.Queue(maxsize=self._q.maxsize)
            self._pid = pid
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="flwr-viz-emitter", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._send_once(max_wait=_BATCH_INTERVAL)
            except Exception:  # pragma: no cover - defensive
                time.sleep(_BATCH_INTERVAL)

    def _drain(self, max_wait: float) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        try:
            batch.append(self._q.get(timeout=max_wait) if max_wait > 0 else self._q.get_nowait())
        except queue.Empty:
            return batch
        while len(batch) < _BATCH_MAX:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send_once(self, max_wait: float) -> None:
        batch = self._drain(max_wait)
        if not batch:
            return
        now = time.monotonic()
        if now < self._next_try:
            self._count_drop(len(batch))  # collector down: drop quietly
            return
        reported = self._dropped_unreported
        if reported:
            role, node, run = self._last_src
            batch.append(
                make_event(
                    "sys.dropped", role=role, node_id=node, run_id=run,
                    data={"count": reported},
                )
            )
        body = json.dumps(batch, default=_json_default, separators=(",", ":")).encode()
        req = urllib.request.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("X-Flwr-Viz-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                resp.read()
            self._dropped_unreported -= reported
            self.sent += len(batch) - (1 if reported else 0)
            self._fail_streak = 0
        except Exception:
            self.failures += 1
            real = len(batch) - (1 if reported else 0)
            self._count_drop(real)
            step = _BACKOFF_STEPS[min(self._fail_streak, len(_BACKOFF_STEPS) - 1)]
            self._fail_streak += 1
            self._next_try = time.monotonic() + step


_EMITTERS: dict[tuple[str, str | None], Emitter] = {}
_REG_LOCK = threading.Lock()


def get_emitter(url: str, token: str | None = None) -> Emitter:
    key = (url, token)
    em = _EMITTERS.get(key)
    if em is None:
        with _REG_LOCK:
            em = _EMITTERS.get(key)
            if em is None:
                em = Emitter(url, token)
                _EMITTERS[key] = em
    return em


@atexit.register
def _flush_all() -> None:  # pragma: no cover - exercised at interpreter exit
    for em in list(_EMITTERS.values()):
        try:
            em.flush(_EXIT_FLUSH_CAP / max(1, len(_EMITTERS)))
        except Exception:
            pass
