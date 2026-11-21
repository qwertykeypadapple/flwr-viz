"""Client-side Flower mod: observes every message a ClientApp handles."""

from __future__ import annotations

import time
from typing import Any, Callable

from .emitter import get_emitter
from .errors import short_exception, short_reason
from .progress import _Current, _enter, _exit
from .schema import fid, make_event
from .settings import resolve
from .stats import metrics_of, named_arrays, summarize, total_bytes

# Types re-declared to avoid importing flwr at import time of this module's
# users who only want the schema; flwr is a hard dependency anyway.
ModFn = Callable[[Any, Any, Callable[[Any, Any], Any]], Any]


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _short_error(exc: BaseException) -> str:
    return short_exception(exc)


def make_viz_mod(
    *,
    url: str | None = None,
    level: str | None = None,
    token: str | None = None,
    layer_names: bool | None = None,
    max_layers: int | None = None,
    progress_hz: float | None = None,
) -> ModFn:
    """Create a Flower mod. Pass it first in ``ClientApp(mods=[...])``.

    Arguments override the ``viz-*`` keys in the run config, which override
    the ``FLWR_VIZ_*`` environment variables.
    """

    def viz_mod(msg: Any, ctxt: Any, call_next: Callable[[Any, Any], Any]) -> Any:
        try:
            s = resolve(
                getattr(ctxt, "run_config", None), url=url, level=level, token=token,
                layer_names=layer_names, max_layers=max_layers, progress_hz=progress_hz,
            )
        except Exception:
            return call_next(msg, ctxt)
        if not s.enabled:
            return call_next(msg, ctxt)

        em = get_emitter(s.url, s.token)
        node_id = _safe(lambda: fid(ctxt.node_id))
        run_id = _safe(lambda: fid(ctxt.run_id))
        md = msg.metadata
        msg_id = _safe(lambda: str(md.message_id))
        mtype = _safe(lambda: str(md.message_type), "?")
        detailed_layers = s.level in ("standard", "detailed")

        def ev(etype: str, data: dict[str, Any]) -> None:
            em.emit(make_event(etype, role="client", node_id=node_id, run_id=run_id,
                               msg_id=msg_id, data=data))

        received: list[tuple[str, Any]] = []
        try:
            received = named_arrays(msg.content) if detailed_layers else []
            ev("client.recv", {
                "mtype": mtype,
                "bytes": total_bytes(msg.content),
                "layers": summarize(received, level=s.level, max_layers=s.max_layers,
                                    layer_names=s.layer_names),
            })
            ev("client.start", {"mtype": mtype})
        except Exception:
            pass

        cur = _Current(em, s, node_id, run_id, msg_id, mtype)
        token_cv = _enter(cur)
        t0 = time.perf_counter()
        try:
            reply = call_next(msg, ctxt)
        except BaseException as exc:
            dur = time.perf_counter() - t0
            _safe(lambda: ev("client.end", {
                "mtype": mtype, "ok": False, "duration_s": dur,
                "error": _short_error(exc), "metrics": {}, "bytes": 0, "layers": [],
            }))
            raise
        finally:
            _exit(cur, token_cv)
        dur = time.perf_counter() - t0

        try:
            if reply.has_error():
                err = reply.error
                ev("client.end", {
                    "mtype": mtype, "ok": False, "duration_s": dur,
                    "error": f"code {err.code}: {short_reason(err.reason or '')}",
                    "metrics": {}, "bytes": 0, "layers": [],
                })
            else:
                out = named_arrays(reply.content) if detailed_layers else []
                ref = dict(received) if received else None
                ev("client.end", {
                    "mtype": mtype, "ok": True, "duration_s": dur,
                    "metrics": metrics_of(reply.content, s.level),
                    "bytes": total_bytes(reply.content),
                    "layers": summarize(out, level=s.level, reference=ref,
                                        max_layers=s.max_layers, layer_names=s.layer_names),
                })
        except Exception:
            pass
        return reply

    viz_mod.__name__ = "viz_mod"
    return viz_mod


viz_mod = make_viz_mod()
