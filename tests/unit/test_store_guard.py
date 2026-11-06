"""The query guard (stores/guard.py) and the rate limit (stores/limits.py): pure, with a fake clock.
The limit tests are the old Orchestra's (tests/unit/test_sources_limits.py), without budgets."""

from __future__ import annotations

import pytest

from orchestra.stores.guard import Refused, conjoined, match_matchers
from orchestra.stores.kinds import kind_of, name_of, render
from orchestra.stores.limits import DailyCap, RateLimit, per_second

MATCH = '{__name__=~"queue_.*", env="prod"}'


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_every_selector_is_kept_within_the_match():
    q = conjoined('sum by (queue) (rate(queue_depth{queue="a"}[5m]))', '{env="prod"}')
    assert q == 'sum by (queue) (rate(queue_depth{queue="a",env="prod"}[5m]))'
    assert conjoined("queue_depth", MATCH) == '{__name__="queue_depth",__name__=~"queue_.*",env="prod"}'
    assert 'env="other",env="prod"' in conjoined('x{env="other"}', '{env="prod"}')    # asks for both: gets nothing


@pytest.mark.parametrize("q", ["{}", '{__name__=~".+"}', '{env="prod"}', 'sum({__name__=~".*"})', ""])
def test_a_selector_needs_a_matcher_of_its_own(q):
    with pytest.raises(Refused):
        conjoined(q, '{env="prod"}')


@pytest.mark.parametrize("match", ["{}", '{__name__=~".*"}', '{__name__=~".+"}', "a or b"])
def test_a_match_that_selects_everything_is_refused(match):
    with pytest.raises(Refused):
        match_matchers(match)


def test_a_range_past_a_day_is_refused():
    with pytest.raises(Refused, match="reaches back"):
        conjoined("rate(x[2d])", '{env="prod"}')
    assert conjoined("rate(x[1h])", '{env="prod"}')


def test_kinds_by_glob_then_the_old_defaults():
    kinds = {"*_inflight": "level", "jobs_total": "ignore", "*_seconds": "flow"}
    assert kind_of("http_inflight", kinds)[0] == "level"
    assert kind_of("jobs_total", kinds)[0] == "ignore"          # an exact name over the default
    assert kind_of("busy_seconds", kinds)[0] == "flow"
    assert [kind_of(n, {})[0] for n in ("a_total", "a_count", "a_bucket", "a_sum", "queue_depth")] == \
        ["flow", "flow", "ignore", "ignore", "level"]


def test_names_fill_labels_and_drop_what_is_missing():
    names = {"grpc_server_*": "{service} {metric}", "calls_total": "{method} calls ({status})"}
    assert name_of(names, "grpc_server_handled_total", {"service": "pay"}) == "pay grpc_server_handled_total"
    assert name_of(names, "calls_total", {"method": "Get"}) == "Get calls"
    assert name_of(names, "other", {}) is None
    assert render("{a} [{b}]", "m", {"a": "x"}) == "x"


def test_rates():
    assert per_second("1/s") == 1 and per_second("30/m") == 0.5 and per_second("2") == 2


def test_the_bucket_lets_a_burst_through_then_paces_at_its_rate():
    c = Clock()
    rl = RateLimit(5, 10, sleep=c.sleep, clock=c)
    for _ in range(10):
        rl.take()
    assert c.slept == []
    for _ in range(5):
        rl.take()
    assert sum(c.slept) == pytest.approx(1.0)


def test_two_processes_share_one_bucket_through_its_file(tmp_path):
    c = Clock()
    a = RateLimit(1, 2, tmp_path / "s.rate", sleep=c.sleep, clock=c)
    b = RateLimit(1, 2, tmp_path / "s.rate", sleep=c.sleep, clock=c)
    a.take()
    b.take()
    a.take()
    assert sum(c.slept) == pytest.approx(1.0)


def test_the_daily_cap_stops_requests_until_the_next_day(tmp_path):
    c = Clock()
    rl = RateLimit(100, 100, tmp_path / "s.rate", per_day=3, sleep=c.sleep, clock=c)
    for _ in range(3):
        rl.take()
    with pytest.raises(DailyCap, match="made its 3 requests today"):
        rl.take()
    c.t += 86400
    rl.take()
    assert rl.today() == 1
