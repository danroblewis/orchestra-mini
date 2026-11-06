"""Merging the config's layers (SPEC §1): maps key by key; a list of named entries by ``name``, an
overlay entry replacing the repository's of that name whole, ``{name: x, enabled: false}``
removing it; any other value, the later layer's."""

from __future__ import annotations

from typing import Any

NAMED = ("stores", "tools", "machines")


def _named(base: list, over: list) -> list:
    out = {e["name"]: e for e in base if isinstance(e, dict) and "name" in e}
    loose = [e for e in base if not (isinstance(e, dict) and "name" in e)]
    for e in over:
        if isinstance(e, dict) and "name" in e:
            out[e["name"]] = e
        else:
            loose.append(e)
    return loose + list(out.values())


def merge(base: Any, over: Any, key: str | None = None) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = merge(base[k], v, k) if k in base else v
        return out
    if key in NAMED and isinstance(base, list) and isinstance(over, list):
        return _named(base, over)
    return over


def drop_disabled(raw: dict) -> dict:
    """The merged config without the named entries turned off (``enabled: false``), and without
    ``enabled`` on the others."""
    out = dict(raw)
    for k in NAMED:
        if isinstance(out.get(k), list):
            kept = [e for e in out[k] if not (isinstance(e, dict) and e.get("enabled") is False)]
            out[k] = [{x: y for x, y in e.items() if x != "enabled"} if isinstance(e, dict) else e for e in kept]
    return out
