"""The strategies (discovery/strategies.py): conservation alone without the sd extra; with it,
kalman_lna is handed the stocks the counters do not balance exactly, and what it finds is merged."""

from __future__ import annotations

import time

import pytest
from discovery.synthetic import history

from orchestra import metrics
from orchestra.discovery import run_once, strategies

END = time.time() // 60 * 60


def project_with(make_project, **kw):
    p = make_project()
    metrics.for_project(p).write(history(END, **kw))
    return p


def test_without_the_extra_conservation_alone(make_project, monkeypatch):
    monkeypatch.setattr(strategies, "installed", lambda: False)
    m = run_once(project_with(make_project), now=END)
    assert m.discovery["strategy"] == "conservation" and "not installed" in m.discovery["sdlab"]


def test_exact_balances_leave_nothing_to_explain(make_project):
    if not strategies.installed():
        pytest.skip("the sd extra is not installed")
    m = run_once(project_with(make_project), now=END)
    assert m.discovery["strategy"] == "conservation" and "balance every stock exactly" in m.discovery["sdlab"]


def test_kalman_lna_is_handed_a_stock_its_counters_miss_part_of(make_project):
    if not strategies.installed():
        pytest.skip("the sd extra is not installed")
    m = run_once(project_with(make_project, abandon=0.05), now=END)       # seed 7: the counters still balance it, inexactly
    assert [s.series.name for s in m.stocks] == ["queue_depth"] and m.stocks[0].evidence["exact"] < strategies.FLAG_EXACT
    assert m.discovery["strategy"] == "conservation+kalman_lna"
    assert m.discovery["sdlab"]["handed"] == [m.stocks[0].id]
    assert {f.counter.name for f in m.flows if f.counter} == {"arrivals_total", "served_total"}
    for f in m.flows:
        if f.counter is None:
            assert f.current().method == "sdlab:kalman_lna" and f.evidence["describes"] == "the residual"
    assert not m.check()


def test_residual_or_whole_change_by_size():
    levels = {"q": [5, 7, 6, 8, 7]}
    known = {"q": [0, 2, -1, 2, -1]}                 # the counters explain every move but nothing is left
    known_most = {"q": [0, 2, -1, 1, -1]}            # one move in four unexplained
    assert strategies.explains_residual([{"perStep": 0.25}], levels, known_most)
    assert not strategies.explains_residual([{"perStep": 1.5}, {"perStep": 1.5}], levels, known_most)
    assert strategies.explains_residual([], levels, known)
