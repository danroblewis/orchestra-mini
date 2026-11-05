"""The query guard: what may be asked of another store (the owner: "some queries like ``{}`` might
take down the prometheus servers"). Copied from the old Orchestra's ``sources/console.py``.

- **Within the match, always.** The store's ``match`` (one selector) is conjoined into every
  selector of a query (``conjoined``): its matchers are added even on a label the selector
  constrains itself (``{env="other"}`` under ``env="prod"`` asks for both, and gets nothing).
- **A selector of its own.** Each selector needs a metric name or a label matcher of its own, not
  one the match has already and not one that matches everything (``=~".+"``, ``=~".*"``, ``=""``):
  ``{}`` and ``{__name__=~".+"}`` are refused, and so is a match that selects everything.
- **A bounded window**: no read, range or subquery reaches back more than ``MAX_WINDOW_S``.
- **A bounded result**: at most ``MAX_SERIES`` series (the reader counts each selector's series
  with a bounded ``/api/v1/series`` before it runs a query, ``reader.Reader.bounded``)."""

from __future__ import annotations

import re

from .scope import LABEL, matchers, walk


class StoreError(Exception):
    """A store did not answer, or refused: the message names the server, never a credential."""


class Refused(StoreError):
    """A query that is not sent, with why."""


MAX_WINDOW_S = 86400.0        # a read reaches back a day at most
MAX_SERIES = 500              # series one read returns at most
EMPTY = {"", ".*", ".+", "^.*$", "^.+$", "(.*)", "(.+)"}   # a matcher with one of these values selects (nearly) everything
UNIT_S = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400, "y": 365 * 86400}
DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)")
STRINGS = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`")


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'`":
        body = v[1:-1]
        return body if v[0] == "`" else re.sub(r"\\(.)", r"\1", body)
    return v


def _parsed(name: str | None, block: str | None) -> list[tuple[str, str, str]]:
    """A selector's matchers as (label, op, value), its metric name as ``__name__ = name``."""
    out = [("__name__", "=", name)] if name else []
    for label, text in matchers(block or ""):
        if label == "__name__" and not LABEL.match(text):        # a quoted metric name inside the braces
            out.append(("__name__", "=", _unquote(text)))
            continue
        m = LABEL.match(text)
        out.append((m[1], m[2], _unquote(text[m.end():])))
    return out


def matcher_text(label: str, op: str, value: str) -> str:
    return f'{label}{op}"{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'


def match_matchers(match: str) -> list[tuple[str, str, str]]:
    """The matchers every query of the store is conjoined with; ``Refused`` when the match is not
    one selector or selects everything."""
    found: list = []
    try:
        walk(match, lambda name, block, head: found.append(_parsed(name, block)) or "")
    except ValueError as e:
        raise Refused(f"the store's match {match!r} is not a selector: {e}") from None
    if len(found) != 1:
        raise Refused(f"the store's match {match!r} is not one selector")
    if not [x for x in found[0] if x[1] in ("=", "=~") and x[2] not in EMPTY]:
        raise Refused(f"the store's match {match!r} selects everything: name a metric or a label")
    return [x for x in found[0] if not (x[0] == "__name__" and x[2] in EMPTY)]


def _named(x: tuple[str, str, str]) -> bool:
    return x[0] == "__name__" and x[1] == "=" and bool(x[2])


def _own(sel: list[tuple[str, str, str]], scope: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """A selector's matchers of its own: a metric name, or a matcher not the match's and not one
    that matches everything."""
    return [x for x in sel if _named(x) or (x not in scope and x[2] not in EMPTY)]


def conjoined(expr: str, match: str) -> str:
    """``expr`` with the store's match conjoined into every selector; ``Refused`` with the reason
    for a selector with no matcher of its own, or a range longer than ``MAX_WINDOW_S``."""
    scope = match_matchers(match)
    named = any(x[0] == "__name__" for x in scope)

    def one(name, block, head):
        written = (head + "{" + block + "}") if block is not None else head
        try:
            sel = _parsed(name, block)
        except ValueError as e:
            raise Refused(f"{written.strip()}: not a selector ({e})") from None
        if not _own(sel, scope):
            raise Refused(f"{written.strip() or '{}'}: a selector needs a metric name or a label matcher of its own beyond "
                          f"the store's match ({match}); a query never selects every series of the store")
        mine = [matcher_text(*x) for x in sel if not (name and x == ("__name__", "=", name))]
        extra = [matcher_text(*x) for x in scope if x not in sel]       # a matcher written twice is asked once
        if name and not named:
            return name + ("{" + ",".join(mine + extra) + "}" if mine or extra else "")
        return "{" + ",".join(([matcher_text("__name__", "=", name)] if name else []) + mine + extra) + "}"
    if not expr.strip():
        raise Refused("an empty query")
    _ranges(expr)
    return walk(expr, one)


def _ranges(expr: str) -> None:
    """``Refused`` for a range or a subquery (``x[30d]``, ``x[7d:1h]``) longer than ``MAX_WINDOW_S``."""
    for block in re.findall(r"\[([^\]]*)\]", STRINGS.sub('""', expr)):
        secs = sum(float(n) * UNIT_S[u] for n, u in DURATION.findall(block.split(":")[0]))
        if secs > MAX_WINDOW_S:
            raise Refused(f"[{block}] reads {secs / 3600:g} h: a read reaches back {MAX_WINDOW_S / 3600:g} h at most")


def window(start: float, end: float) -> None:
    if end - start > MAX_WINDOW_S:
        raise Refused(f"a window of {(end - start) / 3600:g} h: a read covers {MAX_WINDOW_S / 3600:g} h at most")


def selectors(expr: str) -> list[str]:
    """The vector selectors of ``expr``, as written."""
    out: list[str] = []

    def one(name, block, head):
        out.append((head + "{" + block + "}") if block is not None else head)
        return out[-1]
    walk(expr, one)
    return out
