"""Turn exceptions and Flower error reasons into one short, useful line.

Only exception types and messages are kept, never tracebacks (they can contain
paths and values from the client's environment).
"""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# a line like "OSError: [Errno 28] No space left on device"
_EXC_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)):\s?(.*)$")
LIMIT = 240


def _line(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {msg}".strip().rstrip(":")


def short_exception(exc: BaseException) -> str:
    """'Outer: msg (caused by Root: msg)' using the exception chain."""
    seen, root = set(), exc
    while True:
        nxt = root.__cause__ or (None if root.__suppress_context__ else root.__context__)
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt)); root = nxt
    out = _line(exc)
    if root is not exc:
        out += f" (caused by {_line(root)})"
    return out[:LIMIT]


def short_reason(reason: str) -> str:
    """Pick the meaningful part of a Flower error reason.

    Reasons from the simulation engine are a long string with a wrapped Ray
    traceback. Prefer the first (root) exception line found in it, and the
    last one if different; fall back to the first line.
    """
    text = _ANSI.sub("", reason or "")
    lines = [m for m in (_EXC_LINE.match(l) for l in text.splitlines()) if m]
    if not lines:
        return text.strip().splitlines()[0][:LIMIT] if text.strip() else ""
    noise = ("RayTaskError", "ClientAppException")
    useful = [m for m in lines if not any(n in m.group(1) for n in noise)] or lines
    first = f"{useful[0].group(1).split('.')[-1]}: {useful[0].group(2).strip()}"
    last = f"{useful[-1].group(1).split('.')[-1]}: {useful[-1].group(2).strip()}"
    out = last if first == last else f"{last} (caused by {first})"
    return out[:LIMIT]
