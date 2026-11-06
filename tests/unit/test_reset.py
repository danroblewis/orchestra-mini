"""Clearing the data through its module, ``python -m orchestra.reset`` (what ``orchestra chat`` runs)."""

import json
import os
import subprocess
import sys


def test_reset_clears_the_data_and_keeps_the_config(tmp_path):
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    state = home / "repo"
    (state / "logs").mkdir(parents=True)
    for f in ("events.jsonl", "metrics.jsonl", "items.json", "orchestra.yaml", "web.token"):
        (state / f).write_text("x\n" if f != "orchestra.yaml" else "machines: []\n")
    (state / "scripts").mkdir()                       # the owner's: never removed
    (state / "scripts" / "github_events.py").write_text("print()\n")
    (state / "prompts.md").write_text("mine\n")
    env = {**os.environ, "ORCHESTRA_HOME": str(home)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "orchestra.reset", "--repo", str(repo), *a],  # noqa: E731
                                    env=env, capture_output=True, text=True, check=False, timeout=60)
    dry = run()
    assert dry.returncode == 1 and "events.jsonl" in dry.stdout and (state / "events.jsonl").exists()
    done = run("--yes")
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in state.iterdir()) == ["orchestra.yaml", "prompts.md", "scripts", "web.token"]
    assert (state / "scripts" / "github_events.py").read_text() == "print()\n"


def test_reset_is_refused_while_orchestra_run_runs(tmp_path, capsys):
    from orchestra.reset import clear
    (tmp_path / "events.jsonl").write_text("x\n")
    (tmp_path / "endpoint.json").write_text(json.dumps({"pid": os.getpid()}))
    assert clear(tmp_path, True) == 1 and "stop it first" in capsys.readouterr().err and (tmp_path / "events.jsonl").exists()


def test_the_chat_brief_names_the_reset_and_its_list():
    from pathlib import Path

    from orchestra.reset import DATA
    brief = (Path(__file__).resolve().parents[2] / "orchestra" / "setup" / "chat.md").read_text()
    assert "-m orchestra.reset --yes" in brief and all(f"`{d}`" in brief for d in DATA)
