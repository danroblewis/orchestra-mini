"""The pages (orchestra/web): login required (bearer or the session cookie), each page rendering
what it should, read only."""

from __future__ import annotations

import http.client
import json
import threading
import time

import pytest
from discovery.synthetic import history

from orchestra import metrics, runs
from orchestra.discovery import run_once
from orchestra.events import Events, utc_time
from orchestra.web import make_server
from orchestra.web.auth import token

NOW = time.time() // 60 * 60
TRIAGE = """
import json, sys
seen = set()
for line in sys.stdin:
    seen.add(json.loads(line)["subject"])
for s in sorted(seen):
    view = {"subject": s}
    if s == "ABC-5":
        view["waiting_on"] = "bo"
    if s == "ABC-7":
        view["done"] = True
    print(json.dumps(view))
"""


@pytest.fixture
def site(make_project):
    p = make_project({"machines": [{"name": "worker", "command": "true", "on": "tracker.issue.*"},
                                   {"name": "queues", "command": "true", "every": "1m"}],
                      "projections": [{"name": "triage", "match": "ABC-*", "command": "p/triage.py"}],
                      "stores": [{"name": "far", "command": "orchestra-prometheus http://127.0.0.1:9", "match": '{env="prod"}'}]},
                     **{"p/triage.py": TRIAGE})
    ev = Events.for_project(p)
    for i in range(30):
        ev.bus.append({"specversion": "1.0", "id": f"e{i}", "type": "tracker.issue.created", "source": "person.tracker.alice",
                       "subject": f"ABC-{i}", "time": utc_time(NOW - 1000 + i), "data": {"title": f"issue {i}"}})
    ev.bus.append({"specversion": "1.0", "id": "w", "type": "tracker.issue.blocked", "source": "worker", "subject": "ABC-3",
                   "time": utc_time(NOW - 10), "data": {"waiting_on": "owner"}})
    ev.bus.append({"specversion": "1.0", "id": "c", "type": "tracker.issue.closed", "source": "worker", "subject": "ABC-4",
                   "time": utc_time(NOW - 5)})
    ev.bus.append({"specversion": "1.0", "id": "deep", "type": "loop.went.round", "source": "worker",
                   "time": utc_time(NOW - 3), "depth": 9})
    runs.record(p, {"id": "r1", "machine": "worker", "start": utc_time(NOW - 20), "end": utc_time(NOW - 19),
                    "exit": 0, "events": 1, "cost_usd": 0.5})
    metrics.for_project(p).write(history(NOW))
    run_once(p, now=NOW)
    server = make_server(p, "127.0.0.1:0")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield p, server.server_address[1]
    server.shutdown()
    server.server_close()


def ask(port, path, method="GET", headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read().decode()


def login(p, port) -> dict:
    status, h, _ = ask(port, "/login", "POST", {"Content-Type": "application/x-www-form-urlencoded"},
                       f"token={token(p)}&next=/subjects")
    assert status == 303 and h["Location"] == "/subjects" and "HttpOnly" in h["Set-Cookie"]
    return {"Cookie": h["Set-Cookie"].split(";")[0], "Accept": "text/html"}


def test_login_is_required(site):
    p, port = site
    assert ask(port, "/health")[0] == 200
    status, h, _ = ask(port, "/subjects?page=1", headers={"Accept": "text/html"})
    assert status == 303 and h["Location"] == "/login?next=%2Fsubjects%3Fpage%3D1"
    assert ask(port, "/api/subjects")[0] == 401
    assert ask(port, "/core", headers={"Cookie": "orchestra_t=v1.99999999999.forged"})[0] == 401
    assert "Token" in ask(port, "/login")[2]
    assert ask(port, "/api/subjects", headers={"Authorization": f"Bearer {token(p)}"})[0] == 200
    assert ask(port, "/login", "POST", {"Origin": "http://elsewhere.example"}, "token=x")[0] == 403
    for _ in range(5):
        assert ask(port, "/api/subjects", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert ask(port, "/api/subjects", headers={"Authorization": "Bearer wrong"})[0] == 429


def test_each_page_renders(site):
    p, port = site
    h = login(p, port)
    pages = {path: ask(port, path, headers=h) for path in ("/", "/subjects", "/machines", "/dynamics", "/core")}
    assert all(s == 200 for s, _, _ in pages.values()), {k: v[0] for k, v in pages.items()}
    dash = pages["/"][2]
    assert "Waiting on you" in dash and "ABC-3" in dash and "$0.50" in dash and "Recent runs" in dash
    mach = pages["/machines"][2]
    assert "worker" in mach and "person.tracker.alice" in mach and "on tracker.issue.*" in mach and "tracker.issue.created ×30" in mach and "observed" in mach
    dyn = pages["/dynamics"][2]
    assert "queue_depth" in dyn and "× queue_depth_triage per minute" in dyn and "moved by" in dyn and "worker" in dyn and 'id="model"' in dyn
    core = pages["/core"][2]
    for part in ("bus", "metrics store", "store far", "machine queues", "machine worker", "projection latest", "projection triage",
                 "held facts", "discovery"):
        assert part in core
    assert "failing" in core, "the unreachable store says so"
    assert "1 held: deeper than the limit" in core and "loop.went.round deep from worker, depth 9" in core
    assert "last built" in core
    assert ask(port, "/static/graph.js")[0] == 200 and ask(port, "/static/../server.py")[0] == 404


def test_subjects_page_done_hidden_paged_and_waiting(site):
    p, port = site
    h = login(p, port)
    _, _, first = ask(port, "/subjects", headers=h)          # the overview: a table per kind, the newest of each
    assert "28 open subjects" in first and "tracker.issue" in first and "the newest 10" in first and "ABC-4" not in first
    assert "ABC-7" not in first, "done by its first view that says so (triage), not by latest"
    _, _, kind1 = ask(port, "/subjects?kind=tracker.issue", headers=h)   # one kind: all of it, paged
    assert "1-25 of 28 open subjects" in kind1 and "older" in kind1 and "owner" in kind1 and "bo" in kind1
    _, _, second = ask(port, "/subjects?kind=tracker.issue&page=1", headers=h)
    assert "26-28 of 28" in second and "newer" in second
    _, _, closed = ask(port, "/subjects?closed=1&page=0", headers=h)
    assert "30 subjects" in closed and "ABC-4" in closed and "done" in closed
    status, _, one = ask(port, "/subjects?show=ABC-3", headers=h)
    assert status == 200 and "waits on <b>owner</b>" in one and "tracker.issue.blocked" in one and "<h2>Facts</h2>" in one
    status, _, five = ask(port, "/subjects?show=ABC-5", headers=h)
    assert "waits on <b>bo</b>" in five and "<h2>triage</h2>" in five, "the triage view first, in config order"
    assert ask(port, "/subjects?show=NOPE", headers=h)[0] == 404
    data = json.loads(ask(port, "/api/subjects?closed=1", headers=h)[2])
    assert data["total"] == 30


def test_the_pages_are_read_only(site):
    p, port = site
    h = login(p, port)
    assert ask(port, "/subjects", "POST", h, "")[0] == 405
    status, hh, _ = ask(port, "/logout", "POST", h, "")
    assert status == 303 and "Max-Age=0" in hh["Set-Cookie"]


def test_the_owner_asks_for_a_task_and_replies_on_it(site):
    p, port = site
    h = login(p, port)
    assert 'action="/tasks"' in ask(port, "/", headers=h)[2]
    form = h | {"Content-Type": "application/x-www-form-urlencoded"}
    assert ask(port, "/tasks", "POST", {"Content-Type": "application/x-www-form-urlencoded"}, "text=hi")[0] == 401
    assert ask(port, "/tasks", "POST", form | {"Origin": "http://elsewhere.example"}, "text=hi")[0] == 403
    assert ask(port, "/tasks", "POST", form, "text=+++")[0] == 400
    status, hh, _ = ask(port, "/tasks", "POST", form, "text=Add+a+README")
    assert status == 303 and hh["Location"].startswith("/subjects?show=task%2F")
    subject = hh["Location"].split("show=")[1].replace("%2F", "/")
    (asked,) = [f for f in Events.for_project(p).all(0) if f.get("subject") == subject]
    assert asked["type"] == "owner.task.requested" and asked["source"] == "owner" and asked["data"]["text"] == "Add a README"
    page = ask(port, hh["Location"], headers=h)[2]
    assert "Send reply" in page and "Add a README" in page
    assert ask(port, "/tasks", "POST", form, f"text=yes&subject={subject}")[0] == 303
    assert ask(port, "/tasks", "POST", form, "text=yes&subject=ABC-1")[0] == 400
    assert [f["type"] for f in Events.for_project(p).all(0) if f.get("subject") == subject] == ["owner.task.requested", "owner.task.replied"]
