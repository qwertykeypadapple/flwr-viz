"""Versioned event schema shared by the mod, the strategy wrapper, the collector and the UI.

The full specification lives in docs/SCHEMA.md. Keep the two in sync.
"""

from __future__ import annotations

import time
from typing import Any

SCHEMA_VERSION = 1

# type -> role that emits it ("server", "client" or "any")
EVENT_TYPES: dict[str, str] = {
    "run.start": "server",
    "round.start": "server",
    "msg.sent": "server",
    "msg.reply": "server",
    "msg.timeout": "server",
    "agg.start": "server",
    "agg.end": "server",
    "eval.central": "server",
    "round.end": "server",
    "run.end": "server",
    "run.error": "server",
    "client.recv": "client",
    "client.start": "client",
    "client.progress": "client",
    "client.end": "client",
    "sys.dropped": "any",
}

LEVELS = ("off", "minimal", "standard", "detailed")

ENVELOPE_KEYS = ("v", "type", "ts", "run_id", "src", "round", "msg_id", "data")


def fid(x: Any) -> str | None:
    """Flower ids are 64-bit integers; JSON numbers lose precision above 2**53
    in JavaScript, so every id is sent as a decimal string."""
    if x is None or x == "":
        return None
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return str(x)


def make_event(
    etype: str,
    *,
    role: str,
    node_id: int | str | None,
    run_id: int | str | None,
    data: dict[str, Any] | None = None,
    round: int | None = None,  # pylint: disable=redefined-builtin
    msg_id: str | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one event envelope."""
    return {
        "v": SCHEMA_VERSION,
        "type": etype,
        "ts": time.time() if ts is None else ts,
        "run_id": fid(run_id),
        "src": {"role": role, "node_id": fid(node_id)},
        "round": round,
        "msg_id": msg_id or None,
        "data": data or {},
    }


def normalize_event(ev: Any) -> Any:
    """Coerce integer ids (older emitters, hand-written events) to strings in place."""
    if isinstance(ev, dict):
        if isinstance(ev.get("run_id"), int) and not isinstance(ev.get("run_id"), bool):
            ev["run_id"] = str(ev["run_id"])
        src = ev.get("src")
        if isinstance(src, dict) and isinstance(src.get("node_id"), int):
            src["node_id"] = str(src["node_id"])
        d = ev.get("data")
        if isinstance(d, dict):
            for k in ("dst", "from"):
                if isinstance(d.get(k), int) and not isinstance(d.get(k), bool):
                    d[k] = str(d[k])
            for k in ("sampled", "connected"):
                if isinstance(d.get(k), list):
                    d[k] = [str(x) if isinstance(x, int) else x for x in d[k]]
    return ev


def validate_event(ev: Any) -> str | None:
    """Return an error string if ``ev`` is not a valid v1 event, else None.

    Validation is deliberately shallow (envelope only) so that newer minor
    additions to ``data`` never get rejected.
    """
    if not isinstance(ev, dict):
        return "event is not an object"
    if ev.get("v") != SCHEMA_VERSION:
        return f"unsupported schema version {ev.get('v')!r}"
    etype = ev.get("type")
    if etype not in EVENT_TYPES:
        return f"unknown event type {etype!r}"
    if not isinstance(ev.get("ts"), (int, float)):
        return "ts must be a number"
    src = ev.get("src")
    if not isinstance(src, dict) or src.get("role") not in ("server", "client"):
        return "src.role must be 'server' or 'client'"
    expected = EVENT_TYPES[etype]
    if expected != "any" and src["role"] != expected:
        return f"{etype} must come from a {expected}"
    for key, val in (("run_id", ev.get("run_id")), ("src.node_id", src.get("node_id"))):
        if val is not None and not isinstance(val, str):
            return f"{key} must be a decimal string or null"
    if not isinstance(ev.get("data", {}), dict):
        return "data must be an object"
    rnd = ev.get("round")
    if rnd is not None and not isinstance(rnd, int):
        return "round must be an integer or null"
    return None
