"""Summarise a flwr-viz JSONL recording and flag anything that looks wrong.

Usage: python tools/inspect_run.py run.jsonl
"""

from __future__ import annotations

import collections
import json
import sys


def main(path: str) -> int:
    evs = [json.loads(l) for l in open(path) if l.strip()]
    evs.sort(key=lambda e: e.get("seq", 0))
    c = collections.Counter(e["type"] for e in evs)
    problems: list[str] = []
    print(f"{path}: {len(evs)} events")
    print("  types:", dict(c))
    rs = [e for e in evs if e["type"] == "run.start"]
    if rs:
        d = rs[0]["data"]
        print(f"  strategy={d.get('strategy')} rounds={d.get('num_rounds')} level={d.get('level')} "
              f"initial layers={len(d.get('layers') or [])}")
    else:
        problems.append("no run.start")
    clients = {e["src"]["node_id"] for e in evs if e["src"]["role"] == "client"}
    print(f"  clients seen: {len(clients)}")
    mtypes = collections.Counter(e["data"].get("mtype") for e in evs if e["type"] == "msg.sent")
    print("  message types sent:", dict(mtypes))
    phases = collections.Counter((e.get("round"), e["data"].get("phase")) for e in evs if e["type"] == "round.start")
    print(f"  round.start events: {sum(phases.values())} (rounds {sorted({r for r, _ in phases if r is not None})[:12]})")
    sent = {e["msg_id"] for e in evs if e["type"] == "msg.sent"}
    recv = {e["msg_id"] for e in evs if e["type"] == "client.recv"}
    if recv - sent:
        problems.append(f"{len(recv - sent)} client messages with no matching msg.sent")
    nr = sum(1 for e in evs if e["src"]["role"] == "client" and e.get("round") is None)
    if nr:
        problems.append(f"{nr} client events without a round")
    fails = [e for e in evs if e["type"] == "client.end" and not e["data"].get("ok")]
    bad = [e for e in evs if e["type"] == "msg.reply" and not e["data"].get("ok")]
    to = c.get("msg.timeout", 0)
    print(f"  failures: client.end not ok={len(fails)}, error replies={len(bad)}, timeouts={to}")
    for e in fails[:3]:
        print("    ", e["data"].get("error"))
    ends = [e for e in evs if e["type"] == "client.end" and e["data"].get("ok")]
    if ends:
        lay = ends[-1]["data"].get("layers") or []
        num = [l for l in lay if l.get("params")]
        print(f"  last client update: {len(lay)} layers ({len(num)} numeric), metrics={ends[-1]['data'].get('metrics')}")
        for l in lay[:4]:
            print("    ", {k: l.get(k) for k in ("name", "dtype", "params", "bytes", "norm", "delta_norm")})
    aggs = [e for e in evs if e["type"] == "agg.end"]
    for e in aggs[-2:]:
        d = e["data"]
        print(f"  agg.end r{e.get('round')} {d.get('phase')}: metrics={d.get('metrics')} layers={len(d.get('layers') or [])}")
    grid_only = bool(rs) and rs[0]["data"].get("strategy") == "ServerApp without a Strategy"
    if not aggs and grid_only:
        print("  note: grid-only app, its aggregation happens in its own code and is not observable")
    elif not aggs:
        problems.append("no agg.end (no aggregation seen)")
    if not c.get("run.end") and not c.get("run.error"):
        problems.append("no run.end / run.error")
    dropped = sum(e["data"].get("count", 0) for e in evs if e["type"] == "sys.dropped")
    if dropped:
        problems.append(f"{dropped} events dropped by emitters")
    print("  PROBLEMS:" if problems else "  no problems found", *problems, sep="\n    ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
