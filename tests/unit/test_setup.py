"""``orchestra chat``: Claude Code, briefed with the spec and what is already here: the setup brief with
no config yet or one that doesn't load, else the Supervisor's."""

import os
import subprocess
import sys


def test_chat_with_no_config_sets_up(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "claude").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    (fake / "claude").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "ORCHESTRA_HOME": str(tmp_path / "home"),
           "CAPTURE": str(tmp_path / "args"), "MY_SERVICE_TOKEN": "s3cr3t-value"}
    r = subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "chat", "connect", "jira"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    args = (tmp_path / "args").read_text()
    assert "--append-system-prompt" in args and "You are setting up Orchestra" in args
    assert "# SPEC.md" in args and "config: ok (no file yet)" in args and "-m orchestra.reset --yes" in args
    assert "MY_SERVICE_TOKEN" in args and "s3cr3t-value" not in args       # names, never values
    assert args.rstrip().endswith("connect jira")


def test_chat_on_a_config_that_does_not_load_sets_up(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orchestra.yaml").write_text("not_a_key: 1\n")
    env = {**os.environ, "ORCHESTRA_HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "chat", "--print-prompt"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0 and "setting it up" in r.stderr and "not_a_key" in r.stdout
    assert "You are setting up Orchestra" in r.stdout


def test_chat_briefs_the_supervisor_with_what_is_happening_and_can_continue(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orchestra.yaml").write_text("machines:\n  - name: hello\n    command: ./hello.sh\n    every: 1m\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "claude").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    (fake / "claude").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "ORCHESTRA_HOME": str(tmp_path / "home"),
           "CAPTURE": str(tmp_path / "args")}
    r = subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "chat", "--continue"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    args = (tmp_path / "args").read_text()
    assert args.startswith("--continue\n") and "You are this project's Supervisor" in args
    assert "# What is happening now" in args and "# SPEC.md" in args and "command: ./hello.sh" in args
    assert "error: machine hello: ./hello.sh not found" in args, "what orchestra run would say of it"
