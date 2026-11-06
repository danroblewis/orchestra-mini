"""A temporary project: a repository with ``orchestra.yaml`` and its own ``ORCHESTRA_HOME``."""

from __future__ import annotations

import os
import stat
import textwrap

import pytest
import yaml


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORCHESTRA_REPO", raising=False)
    monkeypatch.delenv("ORCHESTRA_SOURCE", raising=False)
    monkeypatch.delenv("ORCHESTRA_MACHINE", raising=False)
    monkeypatch.delenv("ORCHESTRA_BUS_URL", raising=False)
    monkeypatch.delenv("ORCHESTRA_METRICS_URL", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()

    def make(config: dict | None = None, **scripts: str):
        from orchestra.project import load
        (repo / "orchestra.yaml").write_text(yaml.safe_dump({"project": "t", **(config or {})}))
        for name, body in scripts.items():
            script(repo / name, body)
        return load(repo)

    make.repo = repo
    make.home = tmp_path / "home"
    return make


def script(path, body: str) -> None:
    """An executable file (a Python script when it starts with no ``#!``)."""
    body = textwrap.dedent(body).lstrip()
    if not body.startswith("#!"):
        body = f"#!{os.environ.get('PYTHON', __import__('sys').executable)}\n" + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """The fake ``claude`` (tests/fakes.py) first on ``PATH``; the folder it writes what it saw to."""
    from fakes import CLAUDE
    script(tmp_path / "fakebin" / "claude", CLAUDE)
    seen = tmp_path / "fake"
    seen.mkdir()
    monkeypatch.setenv("PATH", f"{tmp_path / 'fakebin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DIR", str(seen))
    return seen
