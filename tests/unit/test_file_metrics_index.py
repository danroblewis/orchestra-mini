"""The file store reads only what was appended since its last read, and reads a file replaced or
rewritten from the start: a reader that asks every few seconds must not rescan the whole file."""

import json

from orchestra.metrics import Point
from orchestra.metrics import file as filestore
from orchestra.metrics.file import FileMetrics

LAB = {"project": "p", "machine": "m"}


def line(t, v, series="level"):
    return json.dumps({"s": series, "l": LAB, "t": t, "v": v}) + "\n"


def test_a_second_read_parses_only_the_lines_appended_since_the_first(tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    path.write_text("".join(line(t, t) for t in range(1000)))
    store = FileMetrics(path)
    assert len(store.raw("level", LAB)) == 1000
    parsed = []
    loads = json.loads
    monkeypatch.setattr(filestore.json, "loads", lambda s: parsed.append(s) or loads(s))
    for _ in range(5):
        store.latest("level", LAB)
    assert parsed == []
    with path.open("a") as f:                     # another process appends
        f.write(line(1000, 7) + line(1001, 8))
    assert store.latest("level", LAB) == (1001, 8)
    assert len(parsed) == 2


def test_a_line_still_being_written_is_read_once_it_is_whole(tmp_path):
    path = tmp_path / "m.jsonl"
    whole = line(1, 1)
    path.write_text(whole + line(2, 2)[:10])
    store = FileMetrics(path)
    assert store.raw("level", LAB) == [(1, 1)]
    with path.open("a") as f:
        f.write(line(2, 2)[10:])
    assert store.raw("level", LAB) == [(1, 1), (2, 2)]


def test_a_file_replaced_or_rewritten_is_read_again(tmp_path):
    path = tmp_path / "m.jsonl"
    store = FileMetrics(path)
    store.write([Point("level", 1.0, 1, LAB), Point("other", 5.0, 1, LAB)])
    assert store.series() and store.raw("level", LAB) == [(1, 1.0)]
    store.delete("other")                          # replaces the file
    assert [n for n, _ in store.series()] == ["level"]
    path.write_text(line(3, 9))                    # rewritten in place, shorter
    assert store.raw("level", LAB) == [(3, 9)]
    path.write_text(line(4, 2) + line(5, 3))       # rewritten in place, longer
    assert store.raw("level", LAB) == [(4, 2), (5, 3)]
    path.unlink()
    assert store.raw("level", LAB) == [] and store.latest("level", LAB) is None


def test_points_written_out_of_order_read_in_time_order_with_ties_in_file_order(tmp_path):
    store = FileMetrics(tmp_path / "m.jsonl")
    store.write([Point("level", 2.0, 20, LAB), Point("level", 1.0, 10, LAB)])
    store.raw("level", LAB)
    store.write([Point("level", 3.0, 10, LAB), Point("level", 4.0, 30, LAB)])
    assert store.raw("level", LAB) == [(10, 1.0), (10, 3.0), (20, 2.0), (30, 4.0)]
    assert store.raw("level", LAB, start=15, end=25) == [(20, 2.0)]
