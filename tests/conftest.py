from __future__ import annotations

import json
from typing import Any

import pytest

from flwr_viz.schema import validate_event


class ListEmitter:
    """Stands in for the network emitter; keeps events in memory."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, ev: dict[str, Any]) -> None:
        # Every event must be valid and JSON-serialisable, as the real emitter needs.
        assert validate_event(ev) is None, (validate_event(ev), ev)
        json.dumps(ev)
        self.events.append(ev)

    def flush(self, timeout: float = 1.0) -> None:  # noqa: ARG002
        pass

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == etype]


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> ListEmitter:
    em = ListEmitter()
    import flwr_viz.mod as mod
    import flwr_viz.strategy as strat

    monkeypatch.setattr(mod, "get_emitter", lambda url, token=None: em)
    monkeypatch.setattr(strat, "get_emitter", lambda url, token=None: em)
    return em


@pytest.fixture(autouse=True)
def task_identity():
    """Flower 1.44 needs a process task identity to construct Messages."""
    from flwr.supercore.task_identity import TaskIdentity

    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 7, 0, 1
    yield
