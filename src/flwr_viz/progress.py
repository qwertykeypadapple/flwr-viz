"""Optional helper to report progress from inside a local training loop."""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any

from .schema import make_event


@dataclass(eq=False)
class _Current:
    emitter: Any
    settings: Any
    node_id: int | None
    run_id: int | None
    msg_id: str | None
    mtype: str
    last_emit: float = 0.0
    last_epoch: Any = None
    pending: dict[str, Any] | None = field(default=None)


_CURRENT: contextvars.ContextVar[_Current | None] = contextvars.ContextVar(
    "flwr_viz_current", default=None
)
# Handlers active in this process. Used as a fallback when report_progress is
# called from another thread (threads don't inherit context variables): if
# exactly one message is being handled, progress is attributed to it.
_ACTIVE: list[_Current] = []


def _enter(cur: _Current) -> contextvars.Token:
    _ACTIVE.append(cur)
    return _CURRENT.set(cur)


def _exit(cur: _Current, token: contextvars.Token) -> None:
    try:
        _ACTIVE.remove(cur)
    except ValueError:
        pass
    _CURRENT.reset(token)


def report_progress(
    *,
    epoch: int | None = None,
    batch: int | None = None,
    total_batches: int | None = None,
    loss: float | None = None,
    **extra: float,
) -> None:
    """Report local-training progress. Cheap, rate-limited, never raises.

    Does nothing when called outside a message handled by ``viz_mod``, so it is
    safe to leave in code that also runs without Flower. Always sends the
    first report of each epoch and the last batch (``batch == total_batches-1``);
    other reports are limited to ``viz-progress-hz`` per node (default 5/s).
    """
    cur = _CURRENT.get()
    if cur is None:
        if len(_ACTIVE) != 1:
            return
        cur = _ACTIVE[0]
    try:
        s = cur.settings
        now = time.monotonic()
        boundary = epoch != cur.last_epoch or (
            total_batches is not None and batch is not None and batch >= total_batches - 1
        )
        min_gap = 1.0 / s.progress_hz if s.progress_hz > 0 else 0.0
        if not boundary and now - cur.last_emit < min_gap:
            return
        cur.last_emit = now
        cur.last_epoch = epoch
        data: dict[str, Any] = {"epoch": epoch, "batch": batch, "total_batches": total_batches}
        if s.level != "minimal":
            data["loss"] = _num(loss)
            data["extra"] = {k: _num(v) for k, v in extra.items() if _num(v) is not None}
        cur.emitter.emit(make_event(
            "client.progress", role="client", node_id=cur.node_id, run_id=cur.run_id,
            msg_id=cur.msg_id, data=data,
        ))
    except Exception:
        pass


def _num(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)  # works for torch/tf/numpy scalars
        return f if f == f and abs(f) != float("inf") else None
    except Exception:
        return None
