"""Replay a recorded run through the browser's reducer (in Node), no federation needed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flwr_viz.collector.server import Collector
from flwr_viz.synth import generate

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tests" / "js" / "replay_check.mjs"
RECORDED = sorted((ROOT / "examples").glob("*/recorded-run.jsonl"))

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not installed")


def _check(path: Path) -> dict:
    p = subprocess.run([node, str(CHECK), str(path)], capture_output=True, text=True, timeout=120)
    out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"failures": [p.stderr]}
    assert p.returncode == 0, out["failures"]
    return out


def test_synthetic_recording_replays(tmp_path):
    c = Collector(tmp_path)
    c.ingest(generate(clients=20, rounds=4, run_id=5), now=1.0)
    c.ingest([], now=10.0)
    for fh in c.files.values():
        fh.close()
    out = _check(next(tmp_path.glob("*.jsonl")))
    assert out["finished"] and out["rounds"] == 4


@pytest.mark.parametrize("path", RECORDED, ids=[p.parent.name for p in RECORDED])
def test_real_flower_recordings_replay(path):
    out = _check(path)
    assert out["finished"] and out["sent"] > 0


TOUR = ROOT / "tests" / "js" / "tour_check.mjs"
ALL_RECORDED = sorted((ROOT / "examples").glob("**/recorded-run*.jsonl"))


@pytest.mark.parametrize("path", ALL_RECORDED, ids=[f"{p.parent.name}/{p.name}" for p in ALL_RECORDED])
def test_learn_mode_tour_is_answerable(path):
    """Every tour question planned from a real run must have exactly one correct answer."""
    p = subprocess.run([node, str(TOUR), str(path)], capture_output=True, text=True, timeout=120)
    out = json.loads(p.stdout)
    assert p.returncode == 0, out["failures"]
    assert len(out["steps"]) >= 6
