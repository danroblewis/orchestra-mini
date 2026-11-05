"""``orchestra run --web`` end to end: a fake Prometheus store holding a queue and its flows, a
metrics command and an events command. While it runs, the login link it prints logs in and every
page answers; afterwards ``orchestra model`` prints the model discovered from the store."""

from __future__ import annotations

import http.client
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from conftest import script
from discovery.synthetic import history
from fake_prometheus import FakePrometheus

NOW = time.time() // 60 * 60


def test_run_with_the_pages(tmp_path):
    fake = FakePrometheus([(p.series, {**p.labels, "env": "prod"}, [(p.ts, p.value)]) for p in history(NOW)], token="tok")
    # one series per name and labels, as a store keeps them
    merged: dict = {}
    for name, labels, pts in fake.dump:
        merged.setdefault((name, tuple(sorted(labels.items()))), (name, labels, []))[2].extend(pts)
    fake.dump = list(merged.values())
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    (repo / "orchestra.yaml").write_text(yaml.safe_dump({
        "project": "webrun",
        "stores": [{"name": "prom", "url": fake.url, "token_env": "PROM_TOKEN", "match": '{env="prod"}', "rate": "50/s"}],
        "metrics": [{"name": "local", "command": "scripts/m.py", "every": "1s"}],
        "events": [{"name": "feed", "command": "scripts/feed.py", "every": "1s"}],
        "discovery": {"every": "2s", "window": "60m"}}))
    script(repo / "scripts" / "m.py", 'print("local_depth 3")')
    script(repo / "scripts" / "feed.py", 'import json; print(json.dumps({"type": "tracker.issue.created", "subject": "X-1", "id": "x1", '
                                         '"data": {"waiting_on": "owner"}}))')
    env = {**os.environ, "ORCHESTRA_HOME": str(home), "PROM_TOKEN": "tok"}
    for k in ("ORCHESTRA_REPO", "ORCHESTRA_PROCESS", "ORCHESTRA_BUS_URL", "ORCHESTRA_METRICS_URL"):
        env.pop(k, None)
    exe = str(Path(sys.executable).parent / "orchestra")
    proc = subprocess.Popen([exe, "run", "--web", "--listen", "127.0.0.1:0", "--seconds", "8"], cwd=repo, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        link = None
        for line in proc.stdout:
            if m := re.search(r"pages: http://127\.0\.0\.1:(\d+)/login#token=(\S+)", line):
                link = (int(m[1]), m[2])
                break
        assert link, proc.stderr.read()
        port, tok = link
        time.sleep(4)                                          # a few rounds of the commands and discovery
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/login", body=f"token={tok}", headers={"Content-Type": "application/x-www-form-urlencoded"})
        r = c.getresponse()
        r.read()
        assert r.status == 303
        cookie = r.getheader("Set-Cookie").split(";")[0]
        bodies = {}
        for path in ("/", "/items", "/machines", "/dynamics", "/core"):
            c.request("GET", path, headers={"Cookie": cookie})
            r = c.getresponse()
            bodies[path] = r.read().decode()
            assert r.status == 200, path
        assert "X-1" in bodies["/"] and "feed" in bodies["/machines"] and "store prom" in bodies["/core"]
        proc.wait(30)
    finally:
        proc.kill()
        fake.close()
    assert proc.returncode == 0
    out = subprocess.run([exe, "model"], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "model webrun" in out.stdout and "stock queue_depth" in out.stdout, out.stdout
    assert "flow served_total" in out.stdout and "flow arrivals_total" in out.stdout
    assert any(q.startswith("sum(increase(") for q in fake.queries) and all('env="prod"' in q for q in fake.queries)
