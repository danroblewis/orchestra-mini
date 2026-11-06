"""``orchestra query check`` (orchestra/logcheck.py): what the log alone shows to be wrong."""

from __future__ import annotations

import time

from orchestra import runs
from orchestra.cli import main
from orchestra.events import Events, utc_time
from orchestra.logcheck import collectors, large, projections, shapes, stamps_now, truncated


def fact(type, data=None, source="poller", subject="A-1", time="2026-11-05T10:00:00.000Z", recorded="2026-11-05T10:00:00.200Z"):
    return {"type": type, "source": source, "subject": subject, "time": time, "recordedtime": recorded, "data": data}


def test_a_source_that_stamps_now(make_project):
    p = make_project({"machines": [{"name": "triage", "command": "x", "on": "a.*"}]})
    found = stamps_now(p, [fact("jira.issue.changed"), fact("jira.issue.changed"),
                           fact("gh.pr.opened", source="gh", time="2026-11-01T00:00:00Z"), fact("gh.pr.opened", source="gh"),
                           fact("triage.real", source="triage"), fact("triage.real", source="triage"),
                           fact("q.depth.observed", source="q"), fact("q.depth.observed", source="q")])
    assert len(found) == 1 and found[0].startswith("source poller:"), "not one with the source's times, a reacting machine, or observations"


def test_large_data_and_a_changed_shape():
    assert "type big.thing.made: 1 facts with data over 4 KB" in large([fact("big.thing.made", {"body": "x" * 5000}),
                                                                        fact("small.thing.made", {"a": 1})])[0]
    old = [fact("jira.issue.changed", {"status": "a", "assignee": "bo"}) for _ in range(3)]
    new = [fact("jira.issue.changed", {"status": "b", "assignee_id": "17"}) for _ in range(2)]
    found = shapes(old + new)
    assert any("key assignee stopped appearing after 3 facts" in x for x in found)
    assert any("key assignee_id newly appeared after 3 facts" in x for x in found)
    assert not shapes([fact("t.x.y", {"a": 1}), fact("t.x.y", {}), fact("t.x.y", {"a": 1}), fact("t.x.y", {})]), "optional: no shape change"


def test_collectors_silent_or_restating(make_project):
    p = make_project({"machines": [{"name": "quiet", "command": "x", "every": "1m"}, {"name": "never", "command": "x", "every": "1m"},
                                   {"name": "restates", "command": "x", "every": "1m"}, {"name": "fine", "command": "x", "every": "1m"}]})
    now = time.time()
    runs.record(p, {"id": "1", "machine": "quiet", "start": utc_time(now - 600), "facts": 0})
    for i in range(3):
        runs.record(p, {"id": f"r{i}", "machine": "restates", "start": utc_time(now - 10 + i), "facts": 2, "restated": 2})
        runs.record(p, {"id": f"f{i}", "machine": "fine", "start": utc_time(now - 10 + i), "facts": 2, "restated": 1})
    found = collectors(p, now)
    assert any(x.startswith("collector quiet: silent since") for x in found)
    assert "collector never: no runs yet" in found
    assert any(x.startswith("collector restates: each of its last 3 runs added only facts the log already had") for x in found)
    assert not any("fine" in x for x in found)


SUBJECTS = 'import json, sys\nfor s in sorted({json.loads(l)["subject"] for l in sys.stdin})CUT:\n    print(json.dumps({"subject": sMORE}))\n'


def projection(cut: str, more: str) -> str:
    return SUBJECTS.replace("CUT", cut).replace("MORE", more)


def test_each_projection_run_twice_over_its_facts(make_project):
    p = make_project({"projections": [
        {"name": "good", "match": "A-*", "command": "p/good.py"},
        {"name": "partial", "match": "A-*", "command": "p/partial.py"},
        {"name": "erring", "match": "A-*", "command": "p/erring.py"},
        {"name": "clock", "match": "A-*", "command": "p/clock.py"},
        {"name": "failing", "match": "A-*", "command": "p/failing.py"},
        {"name": "nothing", "match": "Z-*", "command": "p/good.py"}]}, **{
        "p/good.py": projection("", ""), "p/partial.py": projection("[:1]", ""),
        "p/erring.py": projection("", ', "error": "bad"'), "p/clock.py": "import time\n" + projection("", ', "t": time.time()'),
        "p/failing.py": "import sys\nsys.exit(4)\n"})
    ev = Events.for_project(p)
    for s in ("A-1", "A-2"):
        ev.publish("t.x.made", subject=s)
    found = projections(p, [f | {"seq": i} for i, f in enumerate(ev.read(0)[0], 1)])
    assert not any(x.startswith("projection good:") for x in found)
    assert "projection partial: no view for 1 subjects (such as A-2)" in found
    assert "projection erring: an error view for 2 keys (such as A-1)" in found
    assert any(x.startswith("projection clock: different views the second time") for x in found)
    assert any(x.startswith("projection failing: exited 4") for x in found)
    assert "projection nothing: no fact matches it yet" in found


def test_query_check_says_nothing_to_report_or_each_finding(make_project, capsys):
    make_project()
    assert main(["--repo", str(make_project.repo), "query", "check"]) == 0 and capsys.readouterr().out == "nothing to report\n"
    make_project({"machines": [{"name": "c", "command": "x", "every": "1m"}]})
    assert main(["--repo", str(make_project.repo), "query", "check"]) == 1 and "collector c: no runs yet" in capsys.readouterr().out


def test_truncated_facts_are_reported():
    facts = [{"type": "jira.issue.status_changed", "data": {"truncated": True}}, {"type": "jira.issue.status_changed", "data": {}}]
    assert truncated(facts) == ["type jira.issue.status_changed: 1 facts say truncated: read the list to its end (page it), "
                                "or say what was left out"]


def test_views_built_step_by_step_are_checked_against_the_whole_log(make_project):
    from orchestra.views.builder import Builder
    counting = """import json, sys
n = 0
for line in sys.stdin:
    f = json.loads(line); n += 1
    print(json.dumps({"subject": f["subject"], "n": n}))
"""
    p = make_project({"projections": [{"name": "leaky", "match": "A-*", "command": "p/leaky.py"}]}, **{"p/leaky.py": counting})
    ev = Events.for_project(p)
    ev.publish("t.x.made", subject="A-1")
    Builder(p).build()
    ev.publish("t.x.made", subject="A-2")
    Builder(p).build()
    found = projections(p, [f | {"seq": i} for i, f in enumerate(ev.read(0)[0], 1)])
    assert any(x.startswith("projection leaky: 1 views built step by step differ") for x in found), found


def test_a_projection_whose_script_is_missing_stops_the_run(make_project):
    from orchestra.check import problems
    p = make_project({"projections": [{"name": "gone", "match": "A-*", "command": ".orchestra/projections/gone.py"}]})
    assert ("error", "projection gone: .orchestra/projections/gone.py not found") in problems(p, stores=False)
