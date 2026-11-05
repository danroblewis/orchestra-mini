"""Discovery strategies: each module here has identify(stocks, run, opts) -> result.

    from orchestra.sdlab.discovery.strategies import load
    identify = load("matrix")

A strategy is how an experiment is tried in isolation: add a module, run the bench with
--strategy <name>, and compare with the others (docs/DISCOVERY-LOOP.md).
"""

from __future__ import annotations

import importlib
import pkgutil


def names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__))


def load(name: str):
    return importlib.import_module(f"{__name__}.{name}").identify
