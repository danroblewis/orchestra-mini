"""The Prometheus text exposition format (version 0.0.4), read into samples.

What a script collector prints and what a scrape target serves::

    # HELP disk_free_bytes Free space on the volume.
    # TYPE disk_free_bytes gauge
    disk_free_bytes{volume="/"} 1.2e+11
    files_written_total 42 1761500000000

A line is a sample (``name{label="value",...} value [timestamp in ms]``), a ``# HELP`` or
``# TYPE`` line, another comment, or blank. Label values take the escapes ``\\\\``, ``\\"`` and
``\\n``. A line that is none of these is skipped and reported, never fatal.

Counters, gauges and untyped series are taken as they are. Of a histogram or a summary only its
``_count`` and ``_sum`` are kept, as plain series (both are counters); its buckets
(``_bucket{le=...}``) and quantiles (``{quantile=...}``) are left out, since they are a
distribution's shape and not a measure discovery can use as a stock or a flow. A value that is not
a number (``NaN``, ``+Inf``, ``-Inf``) is left out too: a store cannot hold it. The OpenMetrics
``# EOF`` line is a comment here."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}
SHAPES = {"histogram": ("_bucket", "_created"), "summary": ("", "_created")}   # the suffixes left out of each


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float
    ts_ms: int | None = None          # the timestamp the line gave, in milliseconds


@dataclass
class Parsed:
    samples: list[Sample] = field(default_factory=list)
    bad: list[tuple[int, str]] = field(default_factory=list)       # (line number, why) of lines skipped
    left_out: list[str] = field(default_factory=list)   # names left out (buckets, quantiles, not a number), once each


class _Bad(ValueError):
    pass


def _skip(line: str, i: int) -> int:
    while line[i:i + 1] in (" ", "\t"):
        i += 1
    return i


def _labels(line: str, i: int) -> tuple[dict[str, str], int]:
    """The labels from ``line[i]`` (just after the ``{``) up to the ``}``, and the index after it."""
    out: dict[str, str] = {}
    while True:
        i = _skip(line, i)
        if line[i:i + 1] == "}":
            return out, i + 1
        m = LABEL.match(line, i)
        if not m:
            raise _Bad("bad label name")
        name, i = m.group(), m.end()
        i = _skip(line, i)
        if line[i:i + 1] != "=":
            raise _Bad(f"label {name} has no value")
        i = _skip(line, i + 1)
        if line[i:i + 1] != '"':
            raise _Bad(f"label {name}'s value is not quoted")
        i += 1
        value: list[str] = []
        while True:
            if i >= len(line):
                raise _Bad(f"label {name}'s value is not closed")
            c = line[i]
            if c == "\\":
                if line[i + 1:i + 2] not in ESCAPES:
                    raise _Bad(f"label {name}'s value has a bad escape")
                value.append(ESCAPES[line[i + 1]])
                i += 2
            elif c == '"':
                i += 1
                break
            else:
                value.append(c)
                i += 1
        if name in out:
            raise _Bad(f"label {name} given twice")
        out[name] = "".join(value)
        i = _skip(line, i)
        if line[i:i + 1] == ",":
            i += 1
        elif line[i:i + 1] != "}":
            raise _Bad("labels not separated by commas or not closed")


def _value(text: str) -> float:
    if not re.fullmatch(r"[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|Inf|inf|NaN|nan)", text):
        raise _Bad(f"bad value {text!r}")
    return float(text)


def sample(line: str) -> Sample:
    """One sample line; raises ValueError saying what is wrong with it."""
    m = NAME.match(line)
    if not m:
        raise _Bad("does not start with a metric name")
    name, i = m.group(), m.end()
    labels: dict[str, str] = {}
    if line[i:i + 1] == "{":
        labels, i = _labels(line, i + 1)
    elif line[i:i + 1] not in (" ", "\t"):
        raise _Bad("bad metric name")
    rest = line[i:].split()
    if not 1 <= len(rest) <= 2:
        raise _Bad("expected a value and at most a timestamp after the name")
    ts = None
    if len(rest) == 2:
        if not re.fullmatch(r"-?\d+", rest[1]):
            raise _Bad(f"bad timestamp {rest[1]!r}: whole milliseconds")
        ts = int(rest[1])
    return Sample(name, labels, _value(rest[0]), ts)


def _shape_of(name: str, types: dict[str, str]) -> bool:
    """Whether ``name`` is the part of a histogram or summary that is left out."""
    if "" in SHAPES.get(types.get(name, ""), ()):
        return True
    return any(name.endswith(suffix) and suffix in SHAPES.get(types.get(name[:-len(suffix)], ""), ())
               for suffix in ("_bucket", "_created"))


def parse(text: str) -> Parsed:
    out = Parsed()
    types: dict[str, str] = {}
    left_out: set[str] = set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].split(None, 3)
            if len(parts) >= 3 and parts[0] == "TYPE":
                types[parts[1]] = parts[2].lower()
            continue
        try:
            s = sample(line)
        except ValueError as e:
            out.bad.append((n, str(e)))
            continue
        if _shape_of(s.name, types) or not math.isfinite(s.value):
            if s.name not in left_out:
                left_out.add(s.name)
                out.left_out.append(s.name)
            continue
        out.samples.append(s)
    return out
