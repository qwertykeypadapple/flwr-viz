"""flwr-viz: live 3D visualization for Flower federated learning runs.

Enable it in three places::

    app = ClientApp(mods=[viz_mod])                    # client_app.py
    strategy = instrument(FedAvg(...), run_config=context.run_config)  # server_app.py
    $ flwr-viz serve                                   # shell

Optionally call ``report_progress(epoch=..., batch=..., loss=...)`` inside
your local training loop.
"""

import warnings as _warnings

try:  # tested with Flower 1.44; the Message API it needs arrived before 1.37
    from flwr import __version__ as _flwr_version

    if tuple(int(x) for x in _flwr_version.split(".")[:2]) < (1, 37):
        _warnings.warn(f"flwr-viz is tested with flwr>=1.37; found {_flwr_version}", stacklevel=2)
except Exception:  # pragma: no cover
    pass

from .mod import make_viz_mod, viz_mod
from .progress import report_progress
from .schema import SCHEMA_VERSION
from .strategy import VizStrategy, instrument

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "VizStrategy",
    "instrument",
    "make_viz_mod",
    "report_progress",
    "viz_mod",
]
