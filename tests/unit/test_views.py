"""The one builder of views (orchestra/views): per-subject projections refolded for the subjects
with new facts, rebuilt when their hash changes, whole ones replaced as a set, the last good view
kept on a failure, one build at a time under the lock, the timeout, ``behind``, and the views a run saw."""

from __future__ import annotations

import json
import threading
import time

from orchestra.events import Events
from orchestra.views import Builder, Views, said
from orchestra.views.builder import locked
from orchestra.views.run import hash_of

COUNT = """
import json, sys
seen = {}
for line in sys.stdin:
    f = json.loads(line)
    v = seen.setdefault(f["subject"], {"subject": f["subject"], "n": 0, "seqs": []})
    v["n"] += 1
    v["seqs"].append(f["seq"])
    if f["type"].endswith("waiting"):
        v["waiting_on"] = "owner"
with open("runs.log", "a") as log:
    log.write(" ".join(sorted(seen)) + "\\n")
for v in seen.values():
    print(json.dumps(v))
"""
WHOLE = """
import json, sys
by = {}
for line in sys.stdin:
    f = json.loads(line)
    who = (f.get("data") or {}).get("by")
    if who:
        by.setdefault(who, set()).add(f["subject"])
for who, subjects in by.items():
    print(json.dumps({"key": who, "subjects": sorted(subjects)}))
"""


def project_with(make_project, projections, **scripts):
    return make_project({"projections": projections}, **scripts)


def put(project, type, subject, data=None):
    return Events.for_project(project).publish(type, data, subject=subject, source="src")


def test_per_subject_views_are_refolded_only_for_subjects_with_new_facts(make_project):
    p = project_with(make_project, [{"name": "count", "match": "T-*", "command": "p/count.py"}], **{"p/count.py": COUNT})
    put(p, "t.task.made", "T-1")
    put(p, "t.task.made", "T-2")
    put(p, "other.thing.made", "X-1")
    b = Builder(p)
    state = b.build()["count"]
    assert set(state["views"]) == {"T-1", "T-2"} and state["built_to"] == 3
    put(p, "t.task.waiting", "T-1")
    state = Builder(p).build()["count"]
    assert state["views"]["T-1"]["n"] == 2 and state["views"]["T-1"]["seqs"] == [1, 4], "every fact of the subject, in log order"
    assert (p.repo / "runs.log").read_text().splitlines() == ["T-1 T-2", "T-1"], "the second run only for T-1"
    assert Builder(p).build()["count"]["built_to"] == 4 and len((p.repo / "runs.log").read_text().splitlines()) == 2
    views = Views(p).of("T-1")
    assert set(views) == {"latest", "count"} and views["count"]["behind"] == 0
    assert said(views, "waiting_on") == "owner" and said(Views(p).of("T-2"), "waiting_on") is None


def test_a_changed_projection_is_rebuilt_from_the_log(make_project):
    p = project_with(make_project, [{"name": "count", "match": "T-*", "command": "projections/count.py"}],
                     **{"projections/count.py": "import lib\n" + COUNT, "projections/lib.py": "X = 1\n",
                        "projections/collector.py": "Y = 1\n"})
    put(p, "t.task.made", "T-1")
    Builder(p).build()
    (p.repo / "projections" / "collector.py").write_text("Y = 2\n")       # a script it doesn't import: no rebuild
    assert Builder(p).build()["count"]["hash"] == hash_of(p, p.config.projections[0])
    assert (p.repo / "runs.log").read_text().splitlines() == ["T-1"]
    (p.repo / "projections" / "lib.py").write_text("X = 2\n")             # a module it imports changed: its hash did
    put(p, "t.task.made", "T-2")
    Builder(p).build()
    assert (p.repo / "runs.log").read_text().splitlines() == ["T-1", "T-1 T-2"], "rebuilt: every subject again"
    Builder(p).build(rebuild={"count"})
    assert (p.repo / "runs.log").read_text().splitlines()[-1] == "T-1 T-2"


def test_a_whole_projection_replaces_its_views_as_a_set(make_project):
    p = project_with(make_project, [{"name": "people", "match": ["T-*", "pr/*"], "types": ["*.reviewed"], "whole": True,
                                     "command": "./whole.py"}], **{"whole.py": WHOLE})
    put(p, "t.task.reviewed", "T-1", {"by": "alice"})
    put(p, "gh.pr.reviewed", "pr/1", {"by": "bo"})
    put(p, "gh.pr.opened", "pr/2", {"by": "carol"})                        # not one of its types
    assert set(Builder(p).build()["people"]["views"]) == {"alice", "bo"}
    put(p, "t.task.reviewed", "T-2", {"by": "alice"})
    p2 = make_project({"projections": [{"name": "people", "match": "T-*", "types": ["*.reviewed"], "whole": True,
                                        "command": "./whole.py"}]})
    state = Builder(p2).build()["people"]
    assert state["views"] == {"alice": {"key": "alice", "subjects": ["T-1", "T-2"]}}, "bo is gone, not an error"
    assert Views(p2).of("alice")["people"]["behind"] == 0
    (p2.repo / "whole.py").write_text("#!/bin/sh\nexit 2\n")
    put(p2, "t.task.reviewed", "T-3", {"by": "dee"})
    state = Builder(p2).build()["people"]
    assert state["views"]["alice"]["subjects"] == ["T-1", "T-2"] and "exited 2" in state["views"]["alice"]["error"], \
        "a failed build keeps the last set, marked"
    assert "exited 2" in state["build"]["error"]


def test_a_failure_keeps_the_last_good_view_and_says_so(make_project):
    p = project_with(make_project, [{"name": "count", "match": "T-*", "command": "p/count.py"}], **{"p/count.py": COUNT})
    put(p, "t.task.made", "T-1")
    Builder(p).build()
    (p.repo / "p" / "count.py").write_text((p.repo / "p" / "count.py").read_text().replace("for v in seen.values():", "raise SystemExit(3)\nfor v in []:"))
    put(p, "t.task.made", "T-1")
    put(p, "t.task.made", "T-9")
    state = Builder(p).build()["count"]
    one, nine = state["views"]["T-1"], state["views"]["T-9"]
    assert one["n"] == 1 and "exited 3" in one["error"] and one["error_at"], "the last good view, marked"
    assert nine == {"subject": "T-9", "error": one["error"], "error_at": nine["error_at"]}, "no good view yet: never latest's"
    assert state["build"]["errors"]
    script = (p.repo / "p" / "count.py").read_text().replace("raise SystemExit(3)\nfor v in []:", "for v in list(seen.values())[:1]:")
    (p.repo / "p" / "count.py").write_text(script)                                 # prints a line for one subject only
    state = Builder(p).build()["count"]
    printed = [s for s in ("T-1", "T-9") if "error" not in state["views"][s]]
    assert len(printed) == 1 and "printed no line" in state["views"][({"T-1", "T-9"} - set(printed)).pop()]["error"]


def test_a_run_past_the_timeout_is_stopped(make_project):
    p = project_with(make_project, [{"name": "slow", "match": "*", "command": "./slow.py"}],
                     **{"slow.py": "import time, sys\nsys.stdin.read()\ntime.sleep(30)\n"})
    put(p, "t.task.made", "T-1")
    t0 = time.monotonic()
    state = Builder(p, timeout=0.5).build()["slow"]
    assert time.monotonic() - t0 < 10 and "timed out" in state["views"]["T-1"]["error"]


def test_one_build_at_a_time_under_the_lock(make_project):
    p = project_with(make_project, [])
    put(p, "t.task.made", "T-1")
    done = threading.Event()
    with locked(p):
        t = threading.Thread(target=lambda: (Builder(p).build(), done.set()), daemon=True)
        t.start()
        assert not done.wait(0.5), "it waits for the lock"
    assert done.wait(10)


def test_behind_counts_the_facts_past_built_to(make_project):
    p = project_with(make_project, [])
    put(p, "t.task.made", "T-1")
    Builder(p).build()
    put(p, "t.task.made", "T-1")
    put(p, "t.task.made", "T-2")
    assert Views(p, build=False).of("T-1")["latest"]["behind"] == 1
    assert Views(p).of("T-1")["latest"]["behind"] == 0


def test_the_views_a_run_saw_are_kept(make_project, monkeypatch):
    from orchestra.views import seen
    p = make_project()
    monkeypatch.setattr(seen, "KEEP", 3)
    paths = [seen.write(p, f"20261105T10000{i}-x", {"latest": {"subject": "A", "n": i}}) for i in range(5)]
    assert json.loads(open(paths[-1]).read())["latest"]["n"] == 4
    assert sorted(x.name for x in (p.state_dir / "runs").glob("*.views.json")) == [f"20261105T10000{i}-x.views.json" for i in (2, 3, 4)]
    item = json.loads(seen.legacy_item({"latest": {"subject": "A", "fields": {}, "by_type": {"x": 1}, "facts": 4}}))
    assert item == {"subject": "A", "fields": {}, "events": 4}
    assert seen.legacy_item({"latest": {"subject": "A", "fields": {"big": "x" * 40_000}}}) == ""


def test_a_fact_that_lands_while_a_projection_runs_waits_for_the_next_build(make_project):
    """The builder works from one snapshot of the log: a fact read by another thread during a
    build is not marked built, so the next build gives it to the projection."""
    import threading
    import time

    from orchestra.events import Events
    from orchestra.facts import FactLog
    from orchestra.views.builder import Builder
    slow = """import json, sys, time
time.sleep(0.5)
n = {}
for line in sys.stdin:
    f = json.loads(line); n[f["subject"]] = n.get(f["subject"], 0) + 1
for s, k in n.items():
    print(json.dumps({"subject": s, "n": k}))
"""
    p = make_project({"projections": [{"name": "count", "match": "s/*", "command": "p/count.py"}]}, **{"p/count.py": slow})
    ev = Events.for_project(p)
    log = FactLog(ev)
    b = Builder(p, log=log)
    ev.publish("x.happened", {"i": 1}, subject="s/1")
    b.build()
    ev.publish("x.happened", {"i": 2}, subject="s/1")
    t = threading.Thread(target=b.build)
    t.start()
    time.sleep(0.2)
    ev.publish("x.happened", {"i": 3}, subject="s/1")      # lands while the projection runs
    log.update()                                           # read by another thread (the runner's follow)
    t.join()
    assert b.build()["count"]["views"]["s/1"]["n"] == 3


def test_since_is_one_snapshot(tmp_path):
    from orchestra.events import Events
    from orchestra.facts import FactLog
    ev = Events(tmp_path / "e.jsonl")
    ev.publish("a.b.c", subject="x")
    log = FactLog(ev)
    log.update()
    facts, at = log.since(0)
    ev.publish("a.b.c", subject="y")
    log.update()
    assert [f["seq"] for f in facts] == [1] and at == 1, "a reader moving its cursor to at misses nothing"
    assert [f["seq"] for f in log.since(at)[0]] == [2]
