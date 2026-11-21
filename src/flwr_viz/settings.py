"""Resolve plug-in settings from explicit args, the Flower run config, and env vars.

Precedence: explicit argument > run_config key > environment variable > default.
The run config is preferred because Flower ships it with the run to every node,
including SuperNodes on other machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import LEVELS

DEFAULT_URL = "http://127.0.0.1:7878"


@dataclass(frozen=True)
class VizSettings:
    url: str = DEFAULT_URL
    level: str = "standard"
    token: str | None = None
    layer_names: bool = True
    max_layers: int = 64
    progress_hz: float = 5.0

    @property
    def enabled(self) -> bool:
        return self.level != "off"


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def resolve(
    run_config: Mapping[str, Any] | None = None,
    *,
    url: str | None = None,
    level: str | None = None,
    token: str | None = None,
    layer_names: bool | None = None,
    max_layers: int | None = None,
    progress_hz: float | None = None,
) -> VizSettings:
    rc = run_config or {}

    def pick(arg: Any, key: str, env: str, default: Any) -> Any:
        if arg is not None:
            return arg
        if key in rc:
            return rc[key]
        if env in os.environ:
            return os.environ[env]
        return default

    lvl = str(pick(level, "viz-level", "FLWR_VIZ_LEVEL", "standard")).lower()
    if lvl not in LEVELS:
        lvl = "standard"
    if _truthy(os.environ.get("FLWR_VIZ_DISABLE", "0")):
        lvl = "off"
    tok = pick(token, "viz-token", "FLWR_VIZ_TOKEN", None)
    return VizSettings(
        url=str(pick(url, "viz-url", "FLWR_VIZ_URL", DEFAULT_URL)).rstrip("/"),
        level=lvl,
        token=str(tok) if tok else None,
        layer_names=_truthy(pick(layer_names, "viz-layer-names", "FLWR_VIZ_LAYER_NAMES", True)),
        max_layers=int(pick(max_layers, "viz-max-layers", "FLWR_VIZ_MAX_LAYERS", 64)),
        progress_hz=float(pick(progress_hz, "viz-progress-hz", "FLWR_VIZ_PROGRESS_HZ", 5.0)),
    )
