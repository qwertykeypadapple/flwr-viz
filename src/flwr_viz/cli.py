"""Command line: ``flwr-viz serve | replay | synth``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flwr-viz", description="Live 3D view of Flower runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the live collector + web UI")
    s.add_argument("--host", default="127.0.0.1",
                   help="bind address (use 0.0.0.0 for clients on other machines)")
    s.add_argument("--port", type=int, default=7878)
    s.add_argument("--record-dir", default="flwr-viz-runs",
                   help="directory for JSONL recordings")
    s.add_argument("--no-record", action="store_true", help="do not write JSONL files")
    s.add_argument("--token", default=None,
                   help="shared secret emitters must send (viz-token / FLWR_VIZ_TOKEN)")

    r = sub.add_parser("replay", help="serve a recorded JSONL run, no federation needed")
    r.add_argument("file")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=7878)

    g = sub.add_parser("synth", help="stream a SYNTHETIC run to a collector (UI load test)")
    g.add_argument("--url", default="http://127.0.0.1:7878")
    g.add_argument("--clients", type=int, default=50)
    g.add_argument("--rounds", type=int, default=5)
    g.add_argument("--fraction", type=float, default=0.5)
    g.add_argument("--speed", type=float, default=1.0)

    a = p.parse_args(argv)
    if a.cmd in ("serve", "replay"):
        try:
            from .collector.server import Collector, serve  # noqa: F401
        except ImportError:
            print("The collector needs aiohttp: pip install 'flwr-viz[collector]'", file=sys.stderr)
            return 1
    if a.cmd == "serve":
        serve(a.host, a.port, record_dir=None if a.no_record else a.record_dir, token=a.token)
    elif a.cmd == "replay":
        from aiohttp import web

        from .collector.server import Collector

        path = Path(a.file)
        c = Collector(record_dir=path.parent, mode="replay")
        n = c.load_jsonl(path)
        print(f"flwr-viz replay of {path.name}: {n} events "
              f"({c.rejected} rejected) on http://{a.host}:{a.port}")
        web.run_app(c.make_app(), host=a.host, port=a.port, print=None)
    elif a.cmd == "synth":
        from .synth import stream

        n = stream(a.url, speed=a.speed, clients=a.clients, rounds=a.rounds, fraction=a.fraction)
        print(f"streamed {n} synthetic events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
