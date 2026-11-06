"""What a store's metric is and what its series are called: ``kinds``, ``names`` and ``split`` on a
``stores:`` entry. Copied from the old Orchestra's ``sources/kinds.py`` and ``sources/naming.py``.

- ``kinds`` (glob -> ``flow``, ``level`` or ``ignore``): a flow is read as a counter (its
  increments), a level as a gauge, an ignored metric never. Without an entry the defaults are the
  old ones: a histogram's ``_bucket`` and a summary's or histogram's ``_sum`` are not read; a
  ``_total`` or ``_count`` is a counter; everything else is a level.
- ``names`` (glob -> template): ``{label}`` is the series' label, ``{metric}`` the metric's name; a
  label the series lacks is left out, with the brackets and spaces around it ("{method} calls
  ({status})" with no status: "Get calls").
- A key is a name or a glob (``*``, ``?``, ``[…]``); an exact name wins over a pattern, a longer
  pattern over a shorter (``entry``)."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase

KINDS = ("flow", "level", "ignore")
GLOB = set("*?[")
DEFAULTS = {"*_bucket": "ignore", "*_sum": "ignore", "*_total": "flow", "*_count": "flow"}
FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def pattern(key: str) -> bool:
    return bool(GLOB & set(key))


def entry(name: str, table: dict[str, str]) -> str | None:
    """The key of ``table`` that decides for ``name``: the exact name, else the longest pattern
    matching it (ties by the key's own order); None when none does."""
    if name in table and not pattern(name):
        return name
    pats = sorted((k for k in table if pattern(k) and fnmatchcase(name, k)), key=lambda k: (-len(k), k))
    return pats[0] if pats else None


def kind_of(name: str, kinds: dict[str, str]) -> tuple[str, str]:
    """(``flow`` | ``level`` | ``ignore``, why) for a metric of this name."""
    if (k := entry(name, kinds)) is not None:
        return kinds[k], f"kinds: {k}"
    if (k := entry(name, DEFAULTS)) is not None:
        return DEFAULTS[k], f"by default ({k})"
    return "level", "by default (a gauge)"


def render(template: str, metric: str, labels: dict) -> str:
    """The template filled from ``labels``; a label it lacks left out, with the empty brackets and
    extra spaces."""
    out = FIELD.sub(lambda m: metric if m[1] == "metric" else str(labels.get(m[1]) or ""), template)
    out = re.sub(r"\(\s*[,;]?\s*\)|\[\s*\]", "", out)
    out = re.sub(r"\(\s*[,;]\s*|\s*[,;]\s*\)", lambda m: "(" if "(" in m[0] else ")", out)
    return re.sub(r"\s+", " ", out).strip(" ,;:-") or metric


def name_of(names: dict[str, str], metric: str, labels: dict) -> str | None:
    """What a series of ``metric`` with ``labels`` is called by ``names``, or None."""
    k = entry(metric, names)
    return render(names[k], metric, labels) if k is not None else None
