"""``latest``, the built-in projection (orchestra/latest.py): fields by time, ties by seq, by_type, done."""

from __future__ import annotations

from orchestra.latest import view


def fact(type, t, data=None, seq=0):
    return {"type": type, "subject": "T-1", "time": f"2026-11-05T10:00:{t:02d}.000Z", "data": data, "seq": seq}


def test_fields_by_time_and_by_type_kept_together():
    v = view("T-1", [fact("t.task.created", 0, {"title": "a", "owner": "x"}, 1), fact("t.task.edited", 5, {"owner": None}, 2),
                     fact("t.task.edited", 3, {"title": "b"}, 3)])
    assert v["fields"] == {"title": "b"}, "latest by time wins; null removes"
    assert (v["type"], v["first"], v["last"], v["done"], v["facts"]) == ("t.task.edited", fact("", 0)["time"], fact("", 5)["time"],
                                                                         False, 3)
    assert v["by_type"] == {"t.task.created": {"title": "a", "owner": "x"}, "t.task.edited": {"owner": None}}


def test_a_status_change_stays_together_under_its_type():
    v = view("ABC-1", [fact("jira.issue.status_changed", 1, {"from": "To Do", "to": "In Progress", "by": "bo"}, 1),
                       fact("jira.issue.status_changed", 2, {"from": "In Progress", "to": "In Review", "by": "alice"}, 2)])
    assert v["by_type"]["jira.issue.status_changed"] == {"from": "In Progress", "to": "In Review", "by": "alice"}


def test_history_backfilled_later_does_not_become_the_current_state():
    today = fact("jira.issue.observed", 50, {"status": "Done"}, 1)
    old = [fact("jira.issue.status_changed", t, {"to": s}, seq) for seq, (t, s) in enumerate([(1, "Open"), (2, "In Review")], 2)]
    v = view("T-1", [today, *old])
    assert v["fields"]["status"] == "Done" and v["type"] == "jira.issue.observed", "time order, not log order"


def test_ties_keep_seq_order_and_done_reopens():
    a, b = fact("t.task.edited", 1, {"k": 1}, 2), fact("t.task.edited", 1, {"k": 2}, 1)
    assert view("T-1", [a, b])["fields"] == {"k": 1}, "same time: the later seq wins"
    assert view("T-1", [a, fact("t.task.completed", 2)])["done"] is True
    assert view("T-1", [a, fact("t.task.cancelled", 2), fact("t.task.reopened", 3)])["done"] is False


def test_a_subjects_kind_is_its_first_facts_type_less_the_verb_or_its_explicit_kind():
    pr = view("o/r/pull/3", [{"type": "github.pr.opened", "time": "2026-11-05T10:00:00Z", "data": {"title": "x"}},
                             {"type": "github.pr.review_submitted", "time": "2026-11-05T11:00:00Z", "data": {}}])
    assert pr["kind"] == "github.pr"
    assert view("t/1", [{"type": "work.started", "kind": "task", "time": "2026-11-05T10:00:00Z", "data": {}}])["kind"] == "task"
    assert view("x", [{"type": "orchestra.prediction.made", "time": "2026-11-05T10:00:00Z"}])["kind"] is None
