import os
import subprocess
import sys


def test_reset_clears_the_data_and_keeps_the_config(tmp_path):
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    state = home / "repo"
    (state / "logs").mkdir(parents=True)
    for f in ("events.jsonl", "metrics.jsonl", "items.json", "orchestra.yaml", "web.token"):
        (state / f).write_text("x\n" if f != "orchestra.yaml" else "events: []\n")
    env = {**os.environ, "ORCHESTRA_HOME": str(home)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "orchestra", "--repo", str(repo), "reset", *a],  # noqa: E731
                                    env=env, capture_output=True, text=True, check=False, timeout=60)
    dry = run()
    assert dry.returncode == 1 and "events.jsonl" in dry.stdout and (state / "events.jsonl").exists()
    done = run("--yes")
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in state.iterdir()) == ["orchestra.yaml", "web.token"]
