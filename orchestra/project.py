"""A project: its repository, its name, its config and its state directory
(``<home>/<project>/``, beside the overlay: the bus and metrics files, runs, logs, items)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config, build_config, home, layers
from .config.load import find


@dataclass
class Project:
    name: str
    repo: Path
    config: Config
    files: list[Path]

    @property
    def state_dir(self) -> Path:
        return home() / self.name

    def path(self, *parts: str) -> Path:
        p = self.state_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def repo_root(start: Path | None = None) -> Path:
    """``ORCHESTRA_REPO``, else the nearest folder up from ``start`` with a config file or a
    ``.git``, else ``start`` (default: the working directory)."""
    if os.environ.get("ORCHESTRA_REPO"):
        return Path(os.environ["ORCHESTRA_REPO"]).resolve()
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if find(d) or (d / ".git").exists():
            return d
    return here


def load(repo: Path | None = None) -> Project:
    root = Path(repo).resolve() if repo else repo_root()
    name, files, raw = layers(root)
    return Project(name, root, build_config(raw), files)
