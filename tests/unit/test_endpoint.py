"""The events endpoint (orchestra/endpoint.py): POST /events behind the token, once per id."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from orchestra.endpoint import recorded, serve, token
from orchestra.events import Events
from orchestra.lines import Publisher


def post(url, body, tok):
    req = urllib.request.Request(url + "/events", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_post_publishes_once_with_the_token(make_project):
    project = make_project()
    events = Events.for_project(project)
    stop = threading.Event()
    url = serve(project, Publisher(events), stop)
    try:
        assert recorded(project)["url"] == url
        tok = token(project)
        assert (project.state_dir / "events.token").stat().st_mode & 0o077 == 0
        assert post(url, {"type": "ci.build.finished", "id": "b1"}, "wrong")[0] == 401
        assert post(url, {"type": "ci.build.finished", "id": "b1", "subject": "repo#1"}, tok) == (201, {"id": "b1", "duplicate": False})
        assert post(url, {"type": "ci.build.finished", "id": "b1"}, tok) == (200, {"id": "b1", "duplicate": True})
        assert post(url, {"subject": "x"}, tok)[0] == 400
        (ev,), _ = events.read(0)
        assert ev["source"] == "endpoint" and ev["subject"] == "repo#1"
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            assert r.status == 200
    finally:
        stop.set()
