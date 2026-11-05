"""The default metrics store: one JSONL file, one point per line. Fine for one machine and
Orchestra's volumes (thousands of points a day).

Reads are served from an index of the file kept in memory and brought up to date before each
read by parsing only the lines appended since the last one, so a reader that asks often (the web
server's levels and history, every few seconds) costs a ``stat`` and the new lines, not a scan of
the whole file. Other processes append to the same file; a file replaced (``delete`` rewrites
it), truncated or rewritten in place is read again from the start: it is told from an append by
its inode, its size and the last line read still being where it was."""

from __future__ import annotations

import bisect
import json
import threading
from pathlib import Path

from . import MetricsError, Point, matchers_of, matches, refuse_future, step_sample


class _Series:
    __slots__ = ("labels", "ordered", "points")

    def __init__(self, labels: dict[str, str]):
        self.labels = labels
        self.points: list[tuple[float, float]] = []
        self.ordered = True

    def add(self, t: float, v: float) -> None:
        if self.points and t < self.points[-1][0]:
            self.ordered = False
        self.points.append((t, v))

    def sorted_points(self) -> list[tuple[float, float]]:
        if not self.ordered:
            self.points.sort(key=lambda p: p[0])     # stable: equal times keep the file's order
            self.ordered = True
        return self.points


class FileMetrics:
    name = "file"

    def __init__(self, path: Path, future_skew: float | None = None):
        self.path = Path(path)
        self.future_skew = future_skew              # refuse points stamped further ahead (``refuse_future``)
        self._lock = threading.Lock()               # writers in this process
        self._index_lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._ident: tuple[int, int] | None = None
        self._offset = 0
        self._tail = b""                             # the last line read, to tell an append from a rewrite
        self._by_name: dict[str, dict[tuple, _Series]] = {}

    def _appended(self, ident: tuple[int, int], size: int) -> bool:
        """Whether the file is the one read so far with lines added after it."""
        if ident != self._ident or size < self._offset:
            return False
        if not self._tail:
            return True
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset - len(self._tail))
                return f.read(len(self._tail)) == self._tail
        except OSError:
            return False

    def _sync(self) -> None:
        """Bring the index up to date with the file (call with ``_index_lock`` held)."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._reset()
            return
        ident = (st.st_dev, st.st_ino)
        if not self._appended(ident, st.st_size):
            self._reset()
            self._ident = ident
        if st.st_size == self._offset:
            return
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read(st.st_size - self._offset)
        end = data.rfind(b"\n")
        if end < 0:                                  # a line still being written
            return
        lines = data[:end + 1].splitlines()
        for line in lines:
            try:
                r = json.loads(line)
                name, labels, t, v = r["s"], r["l"], r["t"], r["v"]
            except (ValueError, KeyError, TypeError):
                continue
            key = tuple(sorted(labels.items()))
            series = self._by_name.setdefault(name, {})
            s = series.get(key)
            if s is None:
                s = series[key] = _Series(labels)
            s.add(t, v)
        self._offset += end + 1
        self._tail = lines[-1] + b"\n" if lines else b""

    def _select(self, match_name, labels: dict[str, str] | None) -> list[tuple[str, _Series]]:
        """The series whose name ``match_name`` accepts and whose labels include ``labels``
        (call with ``_index_lock`` held)."""
        want = list((labels or {}).items())
        return [(name, s) for name, series in self._by_name.items() if match_name(name)
                for s in series.values() if all(s.labels.get(k) == v for k, v in want)]

    def _records(self):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def write(self, points: list[Point]) -> None:
        points = refuse_future(points, self.name, self.future_skew)
        if not points:
            return
        lines = "".join(json.dumps({"s": p.series, "l": p.labels, "t": p.ts, "v": p.value}) + "\n" for p in points)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(lines)

    def raw(self, series, labels=None, start=None, end=None):
        with self._index_lock:
            self._sync()
            found = self._select(lambda n: n == series, labels)
            if len(found) == 1:
                pts = found[0][1].sorted_points()
            else:
                pts = sorted((p for _, s in found for p in s.points), key=lambda p: p[0])
            return [p for p in pts if (start is None or p[0] >= start) and (end is None or p[0] <= end)]

    def range(self, series, labels, start, end, step):
        return step_sample(self.raw(series, labels, None, end), start, end, step)

    def latest(self, series, labels=None):
        pts = self.raw(series, labels)
        return pts[-1] if pts else None

    def series(self, prefix="", labels=None):
        with self._index_lock:
            self._sync()
            found = self._select(lambda n: n.startswith(prefix), labels)
            return sorted(((n, dict(s.labels)) for n, s in found), key=lambda x: (x[0], sorted(x[1].items())))

    def dump(self, prefix="", labels=None):
        with self._index_lock:
            self._sync()
            found = self._select(lambda n: n.startswith(prefix), labels)
            out = [(n, dict(s.labels), list(s.sorted_points())) for n, s in found]
        return sorted(out, key=lambda g: (g[0], sorted(g[1].items())))

    def query_range(self, q, start, end, step, labels=None):
        from .promql import PromQLError, query_range
        try:
            return query_range(self.dump("", labels), q, start, end, step)
        except PromQLError as e:
            raise MetricsError(str(e)) from e

    def recent(self, prefix, matchers=None, since=0.0, keep=1):
        want = matchers_of(matchers)
        keep = max(1, int(keep))
        out = []
        with self._index_lock:
            self._sync()
            for name, series in self._by_name.items():
                if not name.startswith(prefix):
                    continue
                for s in series.values():
                    if not matches(s.labels, want):
                        continue
                    pts = s.sorted_points()
                    i = bisect.bisect_left(pts, since, key=lambda p: p[0])
                    if i < len(pts):
                        out.append((name, dict(s.labels), list(pts[max(i, len(pts) - keep):])))
        return sorted(out, key=lambda g: (g[0], sorted(g[1].items())))

    def delete(self, series_prefix, labels=None):
        with self._lock:
            keep = [r for r in self._records()
                    if not (r["s"].startswith(series_prefix) and all(r["l"].get(k) == v for k, v in (labels or {}).items()))]
            if self.path.exists():
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
                tmp.replace(self.path)
        with self._index_lock:
            self._reset()

    def close(self) -> None:
        pass
