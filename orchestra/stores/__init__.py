"""Other stores, read in place and never written (SPEC §4): each through its adapter, a program
Orchestra keeps running and asks one request a line (``adapter.py``), with the guard, the caps,
the request log (``engine.py``) and the renaming rules, kinds and split (``read.py``) around it.
Used by ``orchestra query --store``, ``orchestra run``'s checks, the pages and discovery:

- ``query(project, promql, store) -> int``: the query over the last hour, printed;
- ``check(project, store) -> [(ok, line)]``: the store's health and today's requests;
- ``read(project, store, start, end, step)``: the series discovery reads."""

from __future__ import annotations

import json
import sys
import time

from . import reqlog
from .engine import Engine
from .guard import Refused, StoreError
from .read import StoreSeries, read

__all__ = ["Engine", "Refused", "StoreError", "StoreSeries", "check", "query", "read"]


def query(project, promql: str, store: str) -> int:
    s = project.config.store(store)
    if s is None:
        print(f"no store {store}; stores: {', '.join(x.name for x in project.config.stores) or 'none'}", file=sys.stderr)
        return 2
    end = time.time()
    try:
        rows = Engine(project, s).query_range(promql, end - 3600, end, 60)
    except StoreError as e:
        print(f"store {s.name}: {e}", file=sys.stderr)
        return 2
    for labels, pts in rows:
        if pts:
            print(f"{json.dumps(labels)} {pts[-1][1]:g}")
    return 0


def check(project, store) -> list[tuple[bool, str]]:
    engine = Engine(project, store)
    why = engine.health()
    errors = [x for x in reqlog.last(project, store.name, 50) if x.get("error")]
    tail = f"; last error: {errors[-1]['error']}" if errors and why is None else ""
    return [(why is None, f"store {store.name}: " + (why or f"answers; {engine.limiter.today()} of {store.per_day} requests today{tail}"))]
