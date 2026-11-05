from pathlib import Path


def test_the_setup_agents_copy_of_the_spec_is_the_spec():
    root = Path(__file__).resolve().parents[2]
    assert (root / "orchestra" / "setup" / "SPEC.md").read_text() == (root / "SPEC.md").read_text()
