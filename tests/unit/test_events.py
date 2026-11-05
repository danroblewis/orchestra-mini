"""Events.after: every event after an offset, none cut off (orchestra/events.py)."""

from orchestra.events import Events


def test_after_gives_every_new_event_once(tmp_path):
    ev = Events(tmp_path / "events.jsonl")
    for i in range(3):
        ev.emit("bakery.loaf_baked", n=i)
    got, nxt = ev.after(0)
    assert [e["data"]["n"] for e in got] == [0, 1, 2] and nxt == ev.end() == 3
    assert ev.after(nxt) == ([], nxt)
    ev.emit("bakery.loaf_baked", n=3)
    got, nxt = ev.after(nxt)
    assert [e["data"]["n"] for e in got] == [3] and nxt == 4


def test_after_reads_events_that_arrive_while_it_reads(tmp_path):
    ev = Events(tmp_path / "events.jsonl")
    ev.emit("bakery.loaf_baked", n=0)
    read = ev.bus.read
    late = []

    def racing_read(since, limit):              # two more land between end() and the first read
        if not late:
            late.append(1)
            ev.emit("bakery.loaf_baked", n=1)
            ev.emit("bakery.loaf_baked", n=2)
        return read(since, limit)
    ev.bus.read = racing_read
    got, nxt = ev.after(0)
    assert [e["data"]["n"] for e in got] == [0, 1, 2] and nxt == 3


def test_after_reads_through_a_subclass_read(tmp_path):
    class Flat(Events):                          # a library user's view of its own events
        def read(self, since=0, limit=500):
            evs, nxt = super().read(since, limit)
            return [{"type": e["type"], **e["data"]} for e in evs], nxt

    ev = Flat(tmp_path / "events.jsonl")
    ev.emit("bakery.loaf_baked", n=0)
    assert ev.after(0) == ([{"type": "bakery.loaf_baked", "n": 0}], 1)


def test_an_events_time_with_any_offset_is_the_same_instant():
    """A source in another zone stamps ``-08:00``: read as the same instant as its ``Z`` form, so
    every page prints it in the viewer's own zone (they format these seconds with the browser's)."""
    from orchestra.events import event_time

    z = event_time({"time": "2026-11-01T17:19:50Z"})
    assert z == event_time({"time": "2026-11-01T09:19:50-08:00"}) == event_time({"time": "2026-11-01T17:19:50.000+00:00"})
    assert event_time({"time": "not a time"}) is None
