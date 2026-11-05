"""Items (orchestra/items.py): the fold, kept incrementally, rebuilt, and the reducer."""

from __future__ import annotations

from orchestra.events import Events
from orchestra.items import Items, fold


def ev(type, t, data=None, subject="T-1"):
    return {"type": type, "subject": subject, "time": f"2026-11-05T10:00:{t:02d}.000Z", "data": data}


def test_the_fold():
    item = fold("T-1", [ev("t.task.created", 0, {"title": "a", "owner": "x"}), ev("t.task.edited", 5, {"owner": None}),
                        ev("t.task.edited", 3, {"title": "b"})])
    assert item["fields"] == {"title": "b"}, "latest by time wins; null removes"
    assert (item["type"], item["first"], item["last"], item["done"]) == ("t.task.edited", ev("", 0)["time"], ev("", 5)["time"], False)
    assert [e["time"][-7:-5] for e in item["events"]] == ["00", "03", "05"]


def test_ties_keep_bus_order_and_done_reopens():
    a, b = ev("t.task.edited", 1, {"k": 1}), ev("t.task.edited", 1, {"k": 2})
    assert fold("T-1", [a, b])["fields"] == {"k": 2}
    assert fold("T-1", [a, ev("t.task.completed", 2)])["done"] is True
    assert fold("T-1", [a, ev("t.task.cancelled", 2), ev("t.task.reopened", 3)])["done"] is False


def test_kept_incrementally_and_rebuilt(make_project):
    project = make_project()
    events = Events.for_project(project)
    events.publish("t.task.created", {"n": 1}, subject="A", time="2026-11-05T10:00:00Z")
    events.publish("t.task.created", {"n": 1}, subject="B", time="2026-11-05T10:00:01Z")
    events.publish("other.thing.happened")
    items = Items(project, events)
    assert [i["subject"] for i in items.list()] == ["B", "A"]
    events.publish("t.task.closed", {"n": 2}, subject="A", time="2026-11-05T10:00:02Z")
    assert items.update() == ["A"]
    assert [i["subject"] for i in items.list()] == ["B"] and len(items.list(closed=True)) == 2
    again = Items(project, events)
    assert again.at == 4 and again.get("A")["fields"] == {"n": 2}
    (project.state_dir / "items.json").unlink()
    assert Items(project, events).get("A")["done"] is True


def test_a_reducer_replaces_the_fold(make_project):
    project = make_project({"items": [{"match": "task/*", "command": "./r.py"}]}, **{"r.py": """
        import json, sys
        evs = [json.loads(l) for l in sys.stdin]
        print(json.dumps({"type": evs[-1]["type"], "count": len(evs), "done": False}))
    """})
    events = Events.for_project(project)
    events.publish("t.task.made", subject="task/1")
    events.publish("t.task.made", subject="other")
    items = Items(project, events)
    assert items.get("task/1")["count"] == 1 and len(items.get("task/1")["events"]) == 1
    assert "count" not in items.get("other")
