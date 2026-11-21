"""Build the static GitHub Pages site into ./site (or a folder given as argument).

    site/index.html              the presentation (from site_src/)
    site/viewer/                 the real viewer, in static mode (no collector)
    site/viewer/recordings/      the recorded runs + index.json

Usage: python tools/build_site.py [outdir]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "flwr_viz" / "collector" / "static"

# (published name, source recording, title shown in the viewer's "open recording" menu)
RUNS = [
    ("pytorch-fedavg.jsonl", "examples/pytorch-fedavg/recorded-run.jsonl", "PyTorch quickstart · FedAvg · 10 clients"),
    ("sklearn-fedmedian.jsonl", "examples/sklearn-fedmedian/recorded-run.jsonl", "scikit-learn · FedMedian · 6 clients, some failing"),
    ("hub-federated-vae.jsonl", "examples/hub/federated-vae/recorded-run.jsonl", "Flower Hub: federated-vae · FedAvg"),
    ("hub-fed-kmeans.jsonl", "examples/hub/fed-kmeans-flower/recorded-run.jsonl", "Flower Hub: fed-kmeans-flower · no strategy"),
    ("hub-fedmap.jsonl", "examples/hub/nhanes-t2d-fedmap-fl/recorded-run.jsonl", "Flower Hub: nhanes-t2d-fedmap-fl · FedMAP"),
    ("hub-xgboost-failed.jsonl", "examples/hub/quickstart-xgboost/recorded-run-disk-full.jsonl", "Flower Hub: quickstart-xgboost · every client failed"),
]


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    viewer = out / "viewer"
    shutil.copytree(STATIC, viewer / "static", ignore=shutil.ignore_patterns(".DS_Store"))
    # the viewer page, told to read recordings from files instead of a collector
    page = (STATIC / "index.html").read_text()
    cfg = ('<script>window.FLWR_VIZ_STATIC = { dir: "recordings/", manifest: "recordings/index.json", '
           f'default: "{RUNS[0][0]}" }};</script>\n')
    page = page.replace('<script type="importmap">', cfg + '<script type="importmap">', 1)
    (viewer / "index.html").write_text(page)
    (viewer / "static" / "index.html").unlink()
    rec = viewer / "recordings"
    rec.mkdir()
    manifest = []
    for name, src, title in RUNS:
        shutil.copyfile(ROOT / src, rec / name)
        manifest.append({"name": name, "title": title, "bytes": (rec / name).stat().st_size})
    (rec / "index.json").write_text(json.dumps({"runs": manifest}, indent=1))
    for f in (ROOT / "site_src").iterdir():
        dst = out / f.name
        (shutil.copytree if f.is_dir() else shutil.copyfile)(f, dst)
    (out / ".nojekyll").write_text("")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({total / 1e6:.1f} MB, {len(manifest)} recordings)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site")
