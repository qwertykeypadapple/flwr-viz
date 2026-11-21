"""Server-side instrumentation for any Flower ``Strategy``.

``instrument(strategy)`` patches *that instance* (not its class) so that
``start()`` sees a recording Grid proxy, and the configure/aggregate hooks
emit round, message and aggregation events. It returns the same object, so
every attribute and custom method of the user's strategy keeps working.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import re
import time
from typing import Any, Callable, Iterable, Mapping

from flwr.serverapp import Grid

from .emitter import get_emitter
from .errors import short_exception, short_reason
from .schema import fid, make_event
from .settings import VizSettings, resolve
from .stats import arrays_from_record, metric_items, metrics_of, summarize, total_bytes

_ERROR_NAMES = {
    0: "UNKNOWN",
    1: "LOAD_CLIENT_APP_EXCEPTION",
    2: "CLIENT_APP_RAISED_EXCEPTION",
    5: "NODE_UNAVAILABLE",
    6: "MOD_FAILED_PRECONDITION",
    7: "INVALID_FAB",
}


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(text: str, limit: int = 200) -> str:
    """Strip terminal colour codes (Ray adds them) and truncate."""
    return _ANSI.sub("", text or "")[:limit]


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _metric_dict(rec: Any) -> dict[str, Any]:
    return dict(metric_items(rec)) if rec is not None else {}


class _State:
    def __init__(self, settings: VizSettings) -> None:
        self.s = settings
        self.em = get_emitter(settings.url, settings.token)
        self.run_id: str | None = None
        self.round: int | None = None
        self.phase: str | None = None
        self.round_t0: float | None = None
        self.rounds_completed = 0
        # msg_id -> info for messages pushed in the current phase
        self.pending: dict[str, dict[str, Any]] = {}
        self.replied: set[str] = set()
        self.arrays_before: Any = None
        # grid-only mode (a ServerApp that talks to the grid without a Strategy)
        self.standalone = False
        self.paused = False      # a Strategy run is driving this grid right now
        self.label = "ServerApp without a Strategy"
        self.started = False
        self.finished = False
        self.cycle: set[str] = set()
        self.t_start = time.time()

    def emit(self, etype: str, data: dict[str, Any], *, msg_id: str | None = None,
             rnd: int | None | str = "current") -> None:
        try:
            self.em.emit(make_event(
                etype, role="server", node_id=0, run_id=self.run_id, msg_id=msg_id,
                round=self.round if rnd == "current" else rnd, data=data,
            ))
        except Exception:
            pass

    # -- round bookkeeping ----------------------------------------------
    def begin_phase(self, rnd: int, phase: str) -> None:
        if self.round is not None and rnd != self.round:
            self.end_round()
        if self.round != rnd:
            self.round = rnd
            self.round_t0 = time.time()
        self.phase = phase
        self.pending = {}
        self.replied = set()

    def end_round(self) -> None:
        if self.round is None:
            return
        self.emit("round.end", {"duration_s": time.time() - (self.round_t0 or time.time())})
        self.rounds_completed = max(self.rounds_completed, self.round)
        self.round = None
        self.phase = None

    # -- grid-only mode: infer rounds from the message types ----------------
    def auto_round(self, msgs: list[Any], grid: Any) -> None:
        """Each batch of messages is a step. A new round starts when a message
        type repeats (e.g. K-means: cluster, check convergence, cluster, ...)."""
        if not self.standalone or self.paused or not msgs:
            return
        try:
            kinds = sorted({_safe(lambda m=m: str(m.metadata.message_type), "?") for m in msgs})
            key = "+".join(kinds)
            connected = _safe(lambda: [fid(n) for n in grid.get_node_ids()], [])
            if not self.started:
                self.started = True
                self.t_start = time.time()
                self.emit("run.start", {"strategy": self.label, "num_rounds": None, "timeout": None,
                                        "level": self.s.level, "layers": [], "connected": connected}, rnd=None)
                atexit.register(self.finish)
            if self.round is None or key in self.cycle:
                self.begin_phase((self.round or 0) + 1, key)
                self.cycle = {key}
            else:
                self.begin_phase(self.round, key)
                self.cycle.add(key)
            sampled = [_safe(lambda m=m: fid(m.metadata.dst_node_id)) for m in msgs]
            self.emit("round.start", {"phase": key, "connected": connected, "sampled": sampled})
        except Exception:
            pass

    def finish(self, error: str | None = None) -> None:
        if not self.standalone or self.finished or not self.started:
            return
        self.finished = True
        self.end_round()
        data = {"duration_s": time.time() - self.t_start, "rounds_completed": self.rounds_completed}
        if error:
            data["error"] = error
        self.emit("run.error" if error else "run.end", data, rnd=None)
        self.em.flush(1.0)

    # -- messages ---------------------------------------------------------
    def on_sent(self, msgs: list[Any], ids: Iterable[str]) -> None:
        now = time.monotonic()
        for msg, mid in zip(msgs, ids):
            try:
                mid = mid or _safe(lambda: msg.metadata.message_id, "")
                dst = _safe(lambda: fid(msg.metadata.dst_node_id))
                mtype = _safe(lambda: str(msg.metadata.message_type), "?")
                if mid:
                    self.pending[mid] = {"dst": dst, "t": now, "phase": self.phase,
                                         "round": self.round}
                self.emit("msg.sent", {
                    "dst": dst, "mtype": mtype, "phase": self.phase,
                    "bytes": total_bytes(msg.content),
                    "pushed": bool(mid),
                }, msg_id=mid or None)
            except Exception:
                pass

    def on_reply(self, reply: Any) -> None:
        try:
            rid = str(reply.metadata.reply_to_message_id)
            if rid in self.replied:
                return
            self.replied.add(rid)
            info = self.pending.get(rid, {})
            latency = time.monotonic() - info["t"] if "t" in info else None
            data: dict[str, Any] = {
                "from": _safe(lambda: fid(reply.metadata.src_node_id)) or info.get("dst"),
                "latency_s": latency,
                "phase": info.get("phase", self.phase),
            }
            if reply.has_error():
                code = _safe(lambda: int(reply.error.code), 0)
                data.update(ok=False, error_code=_ERROR_NAMES.get(code, str(code)),
                            error_reason=short_reason(_safe(lambda: reply.error.reason, "") or ""),
                            bytes=0)
                if data["from"] in (None, "0"):
                    data["from"] = info.get("dst")
            else:
                data.update(ok=True, bytes=total_bytes(reply.content))
            self.emit("msg.reply", data, msg_id=rid, rnd=info.get("round", self.round))
        except Exception:
            pass

    def settle(self, replies: list[Any]) -> tuple[int, int]:
        """Account for synthetic error replies and infer timeouts. Returns (ok, failed)."""
        ok = failed = 0
        for r in replies:
            self.on_reply(r)  # no-op for replies already seen via pull_messages
            if _safe(lambda: r.has_error(), True):
                failed += 1
            else:
                ok += 1
        now = time.monotonic()
        for mid, info in self.pending.items():
            if mid not in self.replied:
                self.emit("msg.timeout", {"dst": info["dst"], "waited_s": now - info["t"],
                                          "phase": info.get("phase")}, msg_id=mid)
                failed += 1
        return ok, failed


class _GridProxy(Grid):
    """Delegating Grid that reports pushes and replies."""

    def __init__(self, inner: Grid, state: _State) -> None:  # pylint: disable=super-init-not-called
        self._inner = inner
        self._viz = state

    def __getattr__(self, name: str) -> Any:  # anything not overridden
        return getattr(self._inner, name)

    def set_run(self, run: Any) -> None:
        self._inner.set_run(run)

    @property
    def run(self) -> Any:
        return self._inner.run

    def create_message(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.create_message(*args, **kwargs)

    def get_node_ids(self) -> Iterable[int]:
        return self._inner.get_node_ids()

    def get_nodes(self) -> Iterable[Any]:
        return self._inner.get_nodes()

    def push_messages(self, messages: Iterable[Any]) -> Iterable[str]:
        msgs = list(messages)
        self._viz.auto_round(msgs, self._inner)
        ids = list(self._inner.push_messages(msgs))
        self._viz.on_sent(msgs, ids)
        return ids

    def pull_messages(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        replies = list(self._inner.pull_messages(*args, **kwargs))
        for r in replies:
            self._viz.on_reply(r)
        return replies

    def send_and_receive(self, messages: Iterable[Any], *, timeout: float | None = None) -> Iterable[Any]:
        # Same loop as Flower's own Grid.send_and_receive (push, then pull until all
        # replies are in or the timeout passes), but through this proxy so every
        # send and reply is recorded when it actually happens.
        pending = {i for i in self.push_messages(messages) if i}
        end = time.time() + (timeout if timeout is not None else 0.0)
        interval = float(getattr(self._inner, "pull_interval", 0.1) or 0.1)
        out: list[Any] = []
        while timeout is None or time.time() < end:
            got = list(self.pull_messages(pending))
            out.extend(got)
            pending.difference_update({_safe(lambda r=r: r.metadata.reply_to_message_id, "") for r in got})
            if not pending:
                break
            time.sleep(interval)
        st = self._viz
        if pending and st.standalone and not st.paused:
            now = time.monotonic()
            for mid in pending:
                info = st.pending.get(mid, {})
                st.emit("msg.timeout", {"dst": info.get("dst"), "waited_s": now - info.get("t", now),
                                        "phase": info.get("phase")}, msg_id=mid)
        return out


def instrument(
    strategy: Any,
    *,
    run_config: Mapping[str, Any] | None = None,
    url: str | None = None,
    level: str | None = None,
    token: str | None = None,
    layer_names: bool | None = None,
    max_layers: int | None = None,
) -> Any:
    """Instrument a Flower strategy in place and return it.

    Pass ``run_config=context.run_config`` to pick up ``viz-*`` keys from the
    app's pyproject.toml. Calling twice on the same object is a no-op.
    """
    if isinstance(strategy, _GridProxy) or getattr(strategy, "_flwr_viz_state", None) is not None:
        return strategy
    s = resolve(run_config, url=url, level=level, token=token,
                layer_names=layer_names, max_layers=max_layers)
    if not s.enabled:
        return strategy
    if isinstance(strategy, Grid):
        # A ServerApp that drives the grid itself, without a Strategy:
        #   grid = instrument(grid, run_config=context.run_config)
        gst = _State(s)
        gst.standalone = True
        gst.run_id = _safe(lambda: fid(strategy.run.run_id))
        return _GridProxy(strategy, gst)
    st = _State(s)
    strategy._flwr_viz_state = st  # pylint: disable=protected-access
    layers_on = s.level in ("standard", "detailed")

    def layers(record: Any, ref: Any = None) -> list[dict[str, Any]]:
        if not layers_on or record is None:
            return []
        refd = dict(arrays_from_record(ref)) if ref is not None else None
        return summarize(arrays_from_record(record), level=s.level, reference=refd,
                         max_layers=s.max_layers, layer_names=s.layer_names)

    def wrap_configure(name: str, phase: str) -> None:
        orig = getattr(strategy, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def configure(server_round: int, arrays: Any, config: Any, grid: Any) -> Any:
            msgs = list(orig(server_round, arrays, config, grid))
            try:
                st.begin_phase(int(server_round), phase)
                if phase == "train":
                    st.arrays_before = arrays
                connected = _safe(lambda: [fid(n) for n in grid.get_node_ids()], [])
                sampled = [_safe(lambda m=m: fid(m.metadata.dst_node_id)) for m in msgs]
                st.emit("round.start", {"phase": phase, "connected": connected,
                                        "sampled": sampled})
            except Exception:
                pass
            return msgs

        setattr(strategy, name, configure)

    def wrap_aggregate(name: str, phase: str) -> None:
        orig = getattr(strategy, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def aggregate(server_round: int, replies: Iterable[Any]) -> Any:
            replies = list(replies)
            t0 = time.perf_counter()
            try:
                ok, failed = st.settle(replies)
                st.emit("agg.start", {"phase": phase, "n_replies": len(replies),
                                      "n_ok": ok, "n_failed": failed})
            except Exception:
                pass
            result = orig(server_round, replies)
            try:
                dur = time.perf_counter() - t0
                if phase == "train":
                    new_arrays, metrics = (result if isinstance(result, tuple) else (None, None))
                    data = {"phase": phase, "duration_s": dur,
                            "metrics": _metric_dict(metrics) if s.level != "minimal" else {},
                            "updated": new_arrays is not None,
                            "layers": layers(new_arrays, st.arrays_before)}
                else:
                    data = {"phase": phase, "duration_s": dur,
                            "metrics": _metric_dict(result) if s.level != "minimal" else {}}
                st.emit("agg.end", data)
            except Exception:
                pass
            return result

        setattr(strategy, name, aggregate)

    wrap_configure("configure_train", "train")
    wrap_configure("configure_evaluate", "evaluate")
    wrap_aggregate("aggregate_train", "train")
    wrap_aggregate("aggregate_evaluate", "evaluate")

    orig_start = strategy.start
    try:
        sig = inspect.signature(orig_start)
    except (TypeError, ValueError):  # pragma: no cover
        sig = None

    @functools.wraps(orig_start)
    def start(*args: Any, **kwargs: Any) -> Any:
        paused_outer: list[_State] = []
        try:
            if sig is None:
                raise TypeError
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            grid = a.get("grid")
            if isinstance(grid, _GridProxy):      # grid was instrumented too: the strategy takes over
                outer = grid._viz
                outer.paused = True
                paused_outer.append(outer)
                grid = grid._inner
            st.run_id = _safe(lambda: fid(grid.run.run_id))
            if grid is not None:
                a["grid"] = _GridProxy(grid, st)
            efn = a.get("evaluate_fn")
            if efn is not None:
                a["evaluate_fn"] = _wrap_eval(efn)
            st.emit("run.start", {
                "strategy": type(strategy).__name__,
                "num_rounds": a.get("num_rounds"),
                "timeout": a.get("timeout"),
                "level": s.level,
                "layers": layers(a.get("initial_arrays")),
                "connected": _safe(lambda: [fid(n) for n in grid.get_node_ids()], []),
            }, rnd=None)
            args, kwargs = bound.args, bound.kwargs
        except Exception:
            pass
        t0 = time.time()
        try:
            result = orig_start(*args, **kwargs)
        except BaseException as exc:
            st.end_round()
            st.emit("run.error", {"duration_s": time.time() - t0,
                                  "rounds_completed": st.rounds_completed,
                                  "error": short_exception(exc)}, rnd=None)
            st.em.flush(1.0)
            raise
        finally:
            for o in paused_outer:
                o.paused = False
        st.end_round()
        st.emit("run.end", {"duration_s": time.time() - t0,
                            "rounds_completed": st.rounds_completed}, rnd=None)
        st.em.flush(1.0)
        return result

    def _wrap_eval(efn: Callable[[int, Any], Any]) -> Callable[[int, Any], Any]:
        @functools.wraps(efn)
        def evaluate_fn(server_round: int, arrays: Any) -> Any:
            t0 = time.perf_counter()
            res = efn(server_round, arrays)
            try:
                st.emit("eval.central", {
                    "metrics": _metric_dict(res) if s.level != "minimal" else {},
                    "duration_s": time.perf_counter() - t0,
                }, rnd=int(server_round))
            except Exception:
                pass
            return res
        return evaluate_fn

    strategy.start = start
    return strategy


# Convenience alias: VizStrategy(FedAvg(...)) reads naturally too.
def VizStrategy(strategy: Any, **kwargs: Any) -> Any:  # pylint: disable=invalid-name
    return instrument(strategy, **kwargs)


__all__ = ["instrument", "VizStrategy", "metrics_of"]
