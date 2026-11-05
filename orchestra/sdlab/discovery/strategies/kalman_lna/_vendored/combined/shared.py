# Vendored for kalman_lna from sdlab/discovery/strategies/combined/shared.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The main line's model, and lna's, computed once per run for every specialist that starts from
them.

lna, censored, hidden_transit, effect_hats, periodic_inflows, lna_loops, lna_pulses and lna_stack each
begin
with the matrix method's `identify` on the same data (the most expensive shared step); gap_prior and
lna_pulses begin with lna's. While `once()` is open, the name each of them imported it under points
to a cached copy, so it runs once per distinct set of options; each caller gets its own deep copy
(some edit the result in place). The specialists' own code is not touched: if one renames its
import, it simply runs the step itself again.
"""

from __future__ import annotations

import copy
import importlib
import json
from contextlib import contextmanager

from orchestra.sdlab.discovery.identify import VARIANTS
from orchestra.sdlab.discovery.identify import identify as _main_line

# (module, the name it imported the main line's identify under)
USERS = (
    ("lna", "_matrix"),
    ("censored", "main_line"),
    ("hidden_transit", "main_line"),
    ("effect_hats", "matrix_identify"),
    ("periodic_inflows", "matrix_identify"),
    ("lna_loops", "_matrix"),
    ("lna_pulses", "_matrix"),
    ("lna_stack", "_matrix"),
)
# (module, the name it imported lna's identify under)
LNA_USERS = (
    ("gap_prior", "_lna"),
    ("lna_pulses", "_lna"),
)


def _key(stocks: list[dict], opts: dict) -> str | None:
    o = dict(opts or {})
    if o.get("candidates"):
        return None
    o.pop("candidates", None)
    if o.get("variants") is VARIANTS or o.get("variants") == VARIANTS:
        o.pop("variants")
    try:
        return json.dumps([stocks, o], sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return None


def _cached(fn, cache: dict):
    """fn run once per key; a run stopped by a cap (or that raised) is remembered and raised again,
    so a step that ran out of time is not started again by the next specialist."""
    def cached(stocks, run, opts=None):
        key = _key(stocks, opts or {})
        if key is None:
            return fn(stocks, run, opts)
        if key not in cache:
            try:
                cache[key] = fn(stocks, run, dict(opts or {}))
            except BaseException as e:                   # noqa: BLE001
                cache[key] = e
                raise
        if isinstance(cache[key], BaseException):
            raise cache[key]
        return copy.deepcopy(cache[key])
    return cached


def _module(name: str):
    try:
        return importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.{name}")
    except Exception:                                       # noqa: BLE001
        return None


@contextmanager
def once():
    """Within it, the main line and lna run once per run and set of options; yields the cached
    stand-ins by strategy name (lna), for the caller's own calls."""
    lna = _module("lna")
    lna_fn = getattr(lna, "identify", None)
    swaps = [(USERS, _main_line, _cached(_main_line, {}))]
    if lna_fn is not None:
        swaps.append((LNA_USERS, lna_fn, _cached(lna_fn, {})))
    patched = []
    for users, real, cached in swaps:
        for mod, name in users:
            m = _module(mod)
            if m is not None and getattr(m, name, None) is real:
                setattr(m, name, cached)
                patched.append((m, name, real))
    try:
        yield {"lna": swaps[1][2]} if lna_fn is not None else {}
    finally:
        for m, name, real in patched:
            setattr(m, name, real)
