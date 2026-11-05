"""``orchestra setup``: Claude Code, briefed with the spec and what is already here."""

import os
import subprocess
import sys


def test_setup_runs_claude_with_the_brief_and_the_spec(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orchestra.yaml").write_text("events:\n  - name: hello\n    command: ./hello.sh\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "claude").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    (fake / "claude").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "ORCHESTRA_HOME": str(tmp_path / "home"),
           "CAPTURE": str(tmp_path / "args"), "MY_SERVICE_TOKEN": "s3cr3t-value"}
    r = subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "setup", "connect", "jira"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    args = (tmp_path / "args").read_text()
    assert "--append-system-prompt" in args and "You are setting up Orchestra" in args
    assert "# SPEC.md" in args and "command: ./hello.sh" in args
    assert "MY_SERVICE_TOKEN" in args and "s3cr3t-value" not in args       # names, never values
    assert args.rstrip().endswith("connect jira")


def test_setup_starts_on_a_config_that_does_not_load(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orchestra.yaml").write_text("not_a_key: 1\n")
    env = {**os.environ, "ORCHESTRA_HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "setup", "--print-prompt"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0 and "the setup agent will fix it" in r.stderr and "not_a_key" in r.stdout
