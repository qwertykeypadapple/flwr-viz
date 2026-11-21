"""Per-layer summary statistics. Raw values never leave this module."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

NamedArrays = list[tuple[str, Any]]  # (name, flwr.app.Array)


def named_arrays(content: Any) -> NamedArrays:
    """Collect (name, Array) pairs from a RecordDict, keeping references only.

    Names are ``key`` when the message has one ArrayRecord, else ``record/key``.
    """
    out: NamedArrays = []
    try:
        records = list(content.array_records.items())
    except Exception:
        return out
    multi = len(records) > 1
    for rname, rec in records:
        try:
            items = list(rec.items())
        except Exception:
            continue
        for key, arr in items:
            out.append((f"{rname}/{key}" if multi else str(key), arr))
    return out


def arrays_from_record(record: Any) -> NamedArrays:
    """(name, Array) pairs from a single ArrayRecord."""
    try:
        return [(str(k), v) for k, v in record.items()]
    except Exception:
        return []


def _nbytes(arr: Any) -> int:
    try:
        return len(arr.data)
    except Exception:
        return 0


def _to_numeric(arr: Any) -> np.ndarray | None:
    try:
        x = arr.numpy()
    except Exception:
        return None
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "biuf":
        return None
    if x.ndim == 1 and x.dtype == np.uint8:
        # A flat run of bytes is a serialized object (e.g. an XGBoost model), not
        # numbers: its "norm" would be meaningless, so report its size only.
        return None
    return x


def _f(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _l2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(x, dtype=np.float64))))


def _reference_index(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Let a layer be found with or without its record prefix.

    A message may carry several ArrayRecords ("arrays/fc.weight", "icnn/...")
    while the reply carries one ("fc.weight"), or the other way round. Bare
    names are added only when they are unambiguous.
    """
    idx = dict(reference)
    bare: dict[str, list[Any]] = {}
    for name, arr in reference.items():
        if "/" in name:
            bare.setdefault(name.split("/", 1)[1], []).append(arr)
    for key, arrs in bare.items():
        if len(arrs) == 1 and key not in idx:
            idx[key] = arrs[0]
    return idx


def _ref_lookup(idx: Mapping[str, Any], name: str) -> Any:
    if name in idx:
        return idx[name]
    if "/" in name:
        return idx.get(name.split("/", 1)[1])
    return None


def summarize(
    arrays: NamedArrays,
    *,
    level: str = "standard",
    reference: Mapping[str, Any] | None = None,
    max_layers: int = 64,
    layer_names: bool = True,
) -> list[dict[str, Any]]:
    """Return a list of LayerStat dicts (see docs/SCHEMA.md).

    ``reference`` maps layer name -> Array; when given, ``delta_norm`` is the
    L2 norm of (this array - reference array) for layers with equal shapes.
    """
    if level in ("off", "minimal"):
        return []
    stats: list[dict[str, Any]] = []
    ref_idx = _reference_index(reference) if reference is not None else None
    for i, (name, arr) in enumerate(arrays):
        st: dict[str, Any] = {
            "name": name if layer_names else f"layer_{i}",
            "dtype": str(getattr(arr, "dtype", "?")),
            "bytes": _nbytes(arr),
        }
        x = _to_numeric(arr)
        if x is None:
            st["params"] = None
            stats.append(st)
            continue
        st["shape"] = [int(s) for s in x.shape]
        st["params"] = int(x.size)
        st["norm"] = _f(_l2(x)) if x.size else 0.0
        ref = _ref_lookup(ref_idx, name) if ref_idx is not None else None
        if ref is not None:
            y = _to_numeric(ref)
            if y is not None and y.shape == x.shape:
                d = x.astype(np.float64, copy=False) - y.astype(np.float64, copy=False)
                st["delta_norm"] = _f(_l2(d))
        if level == "detailed" and x.size:
            xf = x.astype(np.float64, copy=False)
            st["mean"] = _f(xf.mean())
            st["std"] = _f(xf.std())
            st["min"] = _f(xf.min())
            st["max"] = _f(xf.max())
        stats.append(st)
    return _cap(stats, max_layers)


def _cap(stats: list[dict[str, Any]], max_layers: int) -> list[dict[str, Any]]:
    if max_layers <= 0 or len(stats) <= max_layers:
        return stats
    keep, rest = stats[: max_layers - 1], stats[max_layers - 1 :]
    sq_norm = sum((s.get("norm") or 0.0) ** 2 for s in rest)
    sq_delta = [s.get("delta_norm") for s in rest]
    merged: dict[str, Any] = {
        "name": f"(other {len(rest)} arrays)",
        "dtype": "mixed",
        "bytes": sum(s.get("bytes") or 0 for s in rest),
        "params": sum(s.get("params") or 0 for s in rest),
        "norm": math.sqrt(sq_norm),
        "merged": len(rest),
    }
    if any(d is not None for d in sq_delta):
        merged["delta_norm"] = math.sqrt(sum((d or 0.0) ** 2 for d in sq_delta))
    return keep + [merged]


def total_bytes(content: Any) -> int:
    try:
        return int(sum(rec.count_bytes() for rec in content.values()))
    except Exception:
        return 0


def metrics_of(content: Any, level: str = "standard") -> dict[str, Any]:
    """Copy numeric MetricRecord entries. ConfigRecords are never read."""
    if level in ("off", "minimal"):
        return {}
    out: dict[str, Any] = {}
    try:
        records = list(content.metric_records.items())
    except Exception:
        return out
    multi = len(records) > 1
    for rname, rec in records:
        for k, v in metric_items(rec):
            out[f"{rname}/{k}" if multi else k] = v
    return out


def metric_items(rec: Any) -> Iterable[tuple[str, Any]]:
    try:
        items = list(rec.items())
    except Exception:
        return []
    out = []
    for k, v in items:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fv = _f(v)
            if fv is not None:
                out.append((str(k), int(v) if isinstance(v, int) else fv))
        elif isinstance(v, (list, tuple)) and all(isinstance(e, (int, float)) for e in v):
            out.append((str(k), [float(e) for e in v][:256]))
    return out
