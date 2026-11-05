"""Other stores, read in place and never written (SPEC §1 ``stores:``): any server answering the
Prometheus HTTP query API. The seams ``orchestra query --store``, ``orchestra check`` and
discovery use (``orchestra.hooks``):

- ``query(project, promql, store) -> int``: the query over the last hour, printed;
- ``check(project, store) -> [(ok, line)]``: the store's health and today's requests;
- ``read(project, store, start, end, step)``: the series discovery reads (``read.py``).

Every query goes through the guard (``guard.py``), the store's rate and daily cap (``limits.py``),
and the request log (``reqlog.py``)."""

from __future__ import annotations

import json
import sys
import time

from . import reqlog
from .guard import Refused, StoreError
from .kinds import check as check_kinds
from .read import StoreSeries, read
from .reader import Reader

__all__ = ["Refused", "StoreError", "StoreSeries", "check", "query", "read", "store_named"]


def store_named(project, name: str):
    return next((s for s in project.config.stores if s.name == name), None)


def query(project, promql: str, store: str) -> int:
    s = store_named(project, store)
    if s is None:
        print(f"no store {store}; stores: {', '.join(x.name for x in project.config.stores) or 'none'}", file=sys.stderr)
        return 2
    end = time.time()
    try:
        rows = Reader(project, s).query_range(promql, end - 3600, end, 60)
    except StoreError as e:
        print(f"store {s.name}: {e}", file=sys.stderr)
        return 2
    for labels, pts in rows:
        if pts:
            print(f"{json.dumps(labels)} {pts[-1][1]:g}")
    return 0


def check(project, store) -> list[tuple[bool, str]]:
    out = [(False, f"store {store.name}: {why}") for why in check_kinds(store)]
    reader = Reader(project, store)
    why = reader.health()
    used = reader.http.limiter.today()
    errors = [x for x in reqlog.last(project, store.name, 50) if x.get("error")]
    tail = f"; last error: {errors[-1]['error']}" if errors else ""
    out.append((why is None, f"store {store.name}: " + (why or f"answers; {used} of {store.per_day} requests today{tail}")))
    return out
