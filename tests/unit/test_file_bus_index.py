"""The file bus does not read a long log from its start on every call: a second ``end`` reads only
what was appended, and ``read`` near the end starts at a checkpoint, with the same answers as
reading from the start (offsets are line numbers; malformed lines are skipped but counted)."""

import json
from pathlib import Path

import pytest

from orchestra.bus.file import FileBus


class Counted:
    """A file whose reads are counted, in bytes."""

    def __init__(self, f, tally):
        self.f, self.tally = f, tally

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()

    def __iter__(self):
        for line in self.f:
            self.tally["bytes"] += len(line)
            yield line

    def read(self, n=-1):
        b = self.f.read(n)
        self.tally["bytes"] += len(b)
        return b

    def seek(self, *a):
        return self.f.seek(*a)

    def write(self, s):
        return self.f.write(s)


@pytest.fixture
def big(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    with path.open("w") as f:
        for i in range(200_000):
            f.write(json.dumps({"type": "t", "i": i}) + "\n")
            if i == 100:
                f.write("not json\n")              # skipped, but counted as a line
    tally = {"bytes": 0}
    real = Path.open

    def counted(self, *a, **kw):
        f = real(self, *a, **kw)
        return Counted(f, tally) if self == path else f
    monkeypatch.setattr(Path, "open", counted)
    return path, tally


def test_a_second_end_and_a_read_near_the_end_do_not_read_from_the_start(big):
    path, tally = big
    bus = FileBus(path)
    assert bus.end() == 200_001
    size = path.stat().st_size
    assert tally["bytes"] >= size                   # the first count reads it all, once
    tally["bytes"] = 0
    assert bus.end() == 200_001
    assert tally["bytes"] == 0                      # unchanged: nothing read
    bus.append({"type": "t", "i": 200_000})
    assert bus.end() == 200_002
    assert tally["bytes"] < 1000                    # only the new line, and the bytes before it
    tally["bytes"] = 0
    evs, nxt = bus.read(199_990, limit=100)
    assert nxt == 200_002 and [e["i"] for e in evs] == list(range(199_989, 200_001))
    assert tally["bytes"] < size / 10


def test_the_answers_are_those_of_reading_from_the_start(big):
    path, _ = big
    bus = FileBus(path)
    assert bus.read(0, limit=3) == ([{"type": "t", "i": i} for i in (199_997, 199_998, 199_999)], 200_001)
    evs, _ = bus.read(99, limit=0)
    assert evs == []
    evs, nxt = bus.read(100, limit=200_000)
    assert evs[0]["i"] == 100 and evs[1]["i"] == 101 and nxt == 200_001     # line 102 is the malformed one
    assert bus.read(300_000) == ([], 200_001)


def test_a_file_rewritten_or_still_being_written_is_counted_again(tmp_path):
    path = tmp_path / "events.jsonl"
    bus = FileBus(path)
    assert bus.end() == 0 and bus.read(0) == ([], 0)
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert bus.end() == 2
    path.write_text('{"b": 1}\n{"b": 2}\n{"b": 3}\n')         # rewritten in place, longer
    assert bus.end() == 3 and bus.read(0)[0] == [{"b": 1}, {"b": 2}, {"b": 3}]
    path.write_text('{"c": 1}\n')                             # shrank
    assert bus.end() == 1 and bus.read(0) == ([{"c": 1}], 1)
    with path.open("a") as f:
        f.write('{"c": 2')                                    # a line still being written: not counted yet
    assert bus.end() == 1 and bus.read(0) == ([{"c": 1}], 1)
    with path.open("a") as f:
        f.write('}\n')
    assert bus.end() == 2 and bus.read(1) == ([{"c": 2}], 2)


def test_a_half_written_last_line_is_read_once_its_writer_finishes_it(tmp_path):
    import json
    path = tmp_path / "events.jsonl"
    bus = FileBus(path)
    bus.append({"type": "first", "i": 0})
    second = json.dumps({"type": "second", "i": 1, "pad": "x" * 40}) + "\n"
    with path.open("a") as f:
        f.write(second[:30])                                  # the writer is part-way through it
    assert bus.read(0) == ([{"type": "first", "i": 0}], 1)
    assert bus.end() == 1 and bus.page(0, 10) == ([{"type": "first", "i": 0}], 1)
    with path.open("a") as f:
        f.write(second[30:])
    got, nxt = bus.read(1)
    assert [e["type"] for e in got] == ["second"] and nxt == 2
