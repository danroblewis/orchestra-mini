"""The model store: the one model, with its history (``<state>/model/``). Copied from the old
Orchestra's ``sd/store.py``, without eras, the declared layer and run records.

- **Versions** ``v0001.json``, ``v0002.json``, …: the highest is the current model; past versions
  stay readable. Discovery saves one only when the structure changed (``model.structure``).
- **The latest discovery**, ``discovered.json``, written every round: what the pages and
  ``orchestra model`` read, so nothing rediscovers on request.
- **Records** (``predictions.json``): small files beside them, written atomically."""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

from .model import Model

NAME = re.compile(r"^v(\d{4,})\.json$")


def finite(x):
    """``x`` as strict JSON: infinities and NaN become None."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [finite(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class ModelStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @classmethod
    def for_project(cls, project) -> ModelStore:
        return cls(project.state_dir / "model")

    def _path(self, version: int) -> Path:
        return self.dir / f"v{version:04d}.json"

    def _numbers(self) -> list[int]:
        if not self.dir.exists():
            return []
        return sorted(int(m.group(1)) for p in self.dir.iterdir() if (m := NAME.match(p.name)))

    def _read(self, version: int) -> dict | None:
        p = self._path(version)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _write(self, p: Path, body: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(finite(body), indent=1) + "\n", encoding="utf-8")
        tmp.replace(p)

    def save(self, model: Model, note: str = "") -> int:
        """Save the model as the next version and return its number; one identical to the current
        one adds nothing and returns the current number."""
        if errs := model.check():
            raise ValueError("; ".join(errs))
        body = finite(model.to_json())
        with self.locked("versions"):
            nums = self._numbers()
            if nums and (cur := self._read(nums[-1])) and cur["model"] == body:
                return nums[-1]
            n = (nums[-1] if nums else 0) + 1
            self._write(self._path(n), {"version": n, "saved_at": time.time(), "note": note, "model": body})
            return n

    def load(self, version: int | None = None) -> Model | None:
        nums = self._numbers()
        if not nums:
            return None
        d = self._read(nums[-1] if version is None else version)
        return Model.from_json(d["model"]) if d else None

    def current_version(self) -> int | None:
        nums = self._numbers()
        return nums[-1] if nums else None

    def versions(self) -> list[dict]:
        """Every version, oldest first: its number, when and why it was saved, and its size."""
        out = []
        for n in self._numbers():
            d = self._read(n) or {}
            m = d.get("model", {})
            out.append({"version": n, "saved_at": d.get("saved_at"), "note": d.get("note", ""),
                        "stocks": len(m.get("stocks", [])), "flows": len(m.get("flows", []))})
        return out

    def write_discovery(self, model: Model, at: float | None = None) -> None:
        self._write(self.dir / "discovered.json", {"at": at or time.time(), "model": model.to_json()})

    def latest_discovery(self) -> tuple[float, Model] | None:
        try:
            d = json.loads((self.dir / "discovered.json").read_text(encoding="utf-8"))
            return d["at"], Model.from_json(d["model"])
        except (OSError, ValueError, KeyError):
            return None

    def read_record(self, name: str) -> dict | None:
        try:
            return json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write_record(self, name: str, record: dict) -> None:
        self._write(self.dir / f"{name}.json", record)

    @contextmanager
    def locked(self, name: str):
        """A record held for reading and writing, across processes."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{name}.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
