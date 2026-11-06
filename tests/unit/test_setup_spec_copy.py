from pathlib import Path


def test_the_setup_agents_copy_of_the_spec_is_the_spec():
    root = Path(__file__).resolve().parents[2]
    assert (root / "orchestra" / "setup" / "SPEC.md").read_text() == (root / "SPEC.md").read_text()


def test_the_specs_config_example_loads():
    import re

    import yaml

    from orchestra.config import build_config
    from orchestra.config.load import Loader
    spec = (Path(__file__).resolve().parents[2] / "SPEC.md").read_text()
    raw = yaml.load(re.search(r"```yaml\n(.*?)```", spec, re.S).group(1), Loader=Loader)
    cfg = build_config(raw)
    assert [m.name for m in cfg.machines] == ["github", "ci_triage", "fixer", "reviewer", "deploys"]
    assert cfg.store("chronosphere").command.startswith("orchestra-prometheus https://") and cfg.machine("reviewer").continuous
