"""Event collector: ingest over HTTP, broadcast over WebSocket, record to JSONL."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from ..schema import SCHEMA_VERSION, normalize_event, validate_event

STATIC_DIR = Path(__file__).parent / "static"
HOLD_SECONDS = 1.5  # how long client events wait for the matching msg.sent
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")


class Collector:
    def __init__(
        self,
        record_dir: str | Path | None = "flwr-viz-runs",
        *,
        token: str | None = None,
        max_history: int = 200_000,
        hold_seconds: float = HOLD_SECONDS,
        mode: str = "live",
    ) -> None:
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.mode = mode
        self.hold_seconds = hold_seconds
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.seq = 0
        self.rejected = 0
        self.msg_round: dict[str, int | None] = {}
        self.held: list[tuple[float, dict[str, Any]]] = []
        self.sockets: set[web.WebSocketResponse] = set()
        self.files: dict[Any, Any] = {}
        self.file_names: dict[Any, str] = {}
        self._flusher: asyncio.Task | None = None

    # -- event pipeline -------------------------------------------------
    def _file_for(self, run_id: Any) -> Any:
        if self.record_dir is None:
            return None
        fh = self.files.get(run_id)
        if fh is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"run-{run_id if run_id is not None else 'unknown'}-{stamp}.jsonl"
            fh = open(self.record_dir / name, "a", encoding="utf-8")  # noqa: SIM115
            self.files[run_id] = fh
            self.file_names[run_id] = name
        return fh

    def _commit(self, ev: dict[str, Any], out: list[dict[str, Any]]) -> None:
        self.seq += 1
        ev["seq"] = self.seq
        ev.setdefault("rts", time.time())
        self.history.append(ev)
        fh = self._file_for(ev.get("run_id"))
        if fh is not None:
            fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
        out.append(ev)

    def ingest(self, events: list[Any], now: float | None = None) -> list[dict[str, Any]]:
        """Validate, correlate and record events. Returns events ready to broadcast."""
        now = time.time() if now is None else now
        out: list[dict[str, Any]] = []
        for ev in events:
            err = validate_event(normalize_event(ev))
            if err:
                self.rejected += 1
                continue
            ev["rts"] = now
            if ev["type"] == "msg.sent" and ev.get("msg_id"):
                self.msg_round[ev["msg_id"]] = ev.get("round")
            role = ev["src"]["role"]
            if role == "client" and ev.get("round") is None and ev.get("msg_id"):
                mid = ev["msg_id"]
                if mid in self.msg_round:
                    ev["round"] = self.msg_round[mid]
                else:
                    self.held.append((now, ev))
                    continue
            self._commit(ev, out)
        self._release_held(now, out)
        for fh in self.files.values():
            fh.flush()
        return out

    def _release_held(self, now: float, out: list[dict[str, Any]]) -> None:
        if not self.held:
            return
        keep: list[tuple[float, dict[str, Any]]] = []
        for t, ev in self.held:
            mid = ev.get("msg_id")
            if mid in self.msg_round:
                ev["round"] = self.msg_round[mid]
                self._commit(ev, out)
            elif now - t >= self.hold_seconds:
                self._commit(ev, out)  # never matched: keep it, round unknown
            else:
                keep.append((t, ev))
        self.held = keep

    def load_jsonl(self, path: str | Path) -> int:
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    self.rejected += 1
                    continue
                if validate_event(normalize_event(ev)):
                    self.rejected += 1
                    continue
                self.history.append(ev)
                self.seq = max(self.seq, int(ev.get("seq") or 0))
                n += 1
        return n

    async def broadcast(self, events: list[dict[str, Any]]) -> None:
        if not events or not self.sockets:
            return
        payload = json.dumps({"kind": "events", "events": events}, separators=(",", ":"))
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if self.held:
                out: list[dict[str, Any]] = []
                self._release_held(time.time(), out)
                for fh in self.files.values():
                    fh.flush()
                await self.broadcast(out)

    # -- HTTP handlers --------------------------------------------------
    async def h_ingest(self, request: web.Request) -> web.Response:
        if self.mode != "live":
            return web.json_response({"error": "replay mode"}, status=409)
        if self.token and request.headers.get("X-Flwr-Viz-Token") != self.token:
            return web.json_response({"error": "bad token"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        events = body if isinstance(body, list) else [body]
        out = self.ingest(events)
        await self.broadcast(out)
        return web.json_response({"accepted": len(out)})

    async def h_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(json.dumps({
            "kind": "hello", "mode": self.mode, "schema": SCHEMA_VERSION,
        }))
        hist = list(self.history)
        for i in range(0, len(hist), 5000):
            await ws.send_str(json.dumps({"kind": "history", "events": hist[i:i + 5000],
                                          "done": i + 5000 >= len(hist)},
                                         separators=(",", ":")))
        if not hist:
            await ws.send_str(json.dumps({"kind": "history", "events": [], "done": True}))
        self.sockets.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.sockets.discard(ws)
        return ws

    async def h_runs(self, _request: web.Request) -> web.Response:
        if not self.record_dir or not self.record_dir.exists():
            return web.json_response({"runs": []})
        runs = sorted(
            ({"name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
             for p in self.record_dir.glob("*.jsonl")),
            key=lambda r: r["mtime"], reverse=True,
        )
        return web.json_response({"runs": runs})

    async def h_run_file(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if not self.record_dir or not _SAFE_NAME.match(name):
            raise web.HTTPNotFound()
        path = self.record_dir / name
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "application/x-ndjson"})

    async def h_status(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "mode": self.mode, "events": len(self.history), "seq": self.seq,
            "rejected": self.rejected, "held": len(self.held), "clients": len(self.sockets),
            "recording": {str(k): v for k, v in self.file_names.items()},
        })

    async def h_index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    # -- app ------------------------------------------------------------
    def make_app(self) -> web.Application:
        @web.middleware
        async def no_cache(request: web.Request, handler: Any) -> web.StreamResponse:
            resp = await handler(request)
            if request.path == "/" or request.path.startswith("/static/js/"):
                resp.headers["Cache-Control"] = "no-cache"  # UI updates show on reload
            return resp

        app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[no_cache])
        app.router.add_post("/ingest", self.h_ingest)
        app.router.add_get("/ws", self.h_ws)
        app.router.add_get("/api/status", self.h_status)
        app.router.add_get("/api/runs", self.h_runs)
        app.router.add_get("/api/runs/{name}", self.h_run_file)
        app.router.add_get("/", self.h_index)
        app.router.add_static("/static/", STATIC_DIR)

        async def on_start(_app: web.Application) -> None:
            self._flusher = asyncio.create_task(self._flush_loop())

        async def on_stop(_app: web.Application) -> None:
            if self._flusher:
                self._flusher.cancel()
            for ws in list(self.sockets):
                await ws.close()
            for fh in self.files.values():
                fh.close()

        app.on_startup.append(on_start)
        app.on_shutdown.append(on_stop)
        return app


def serve(host: str = "127.0.0.1", port: int = 7878, **kwargs: Any) -> None:
    collector = Collector(**kwargs)
    replay = kwargs.get("mode") == "replay"
    print(f"flwr-viz collector ({'replay' if replay else 'live'}) on http://{host}:{port}")
    if collector.record_dir and not replay:
        print(f"recording runs to {collector.record_dir.resolve()}")
    web.run_app(collector.make_app(), host=host, port=port, print=None)
