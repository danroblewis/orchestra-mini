"""The config's schema (SPEC §1): plain dataclasses, one per section.

A field's metadata says how it is read: ``required``; ``duration`` (a string like ``30s``, ``5m``,
``2h``, ``1d``, or seconds, kept as seconds); ``names`` (a string or a list, kept as a list).
Every other field has its default. ``load`` builds these from the merged layers."""

from __future__ import annotations

from dataclasses import dataclass, field

REQ = {"required": True}
DUR = {"duration": True}
LIST = {"names": True}


@dataclass
class Store:
    name: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)      # the adapter, kept running, asked on stdin (SPEC §4)
    match: str = field(default="", metadata=REQ)
    names: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    split: list[str] = field(default_factory=list, metadata=LIST)
    rate: str = "1/s"
    per_day: int = 5000
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Machine:
    name: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)
    on: list[str] = field(default_factory=list, metadata=LIST)
    every: float | None = field(default=None, metadata=DUR)
    continuous: bool = False             # kept running, stdin open; else once: one run per event or tick
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Reducer:
    match: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)


@dataclass
class Discovery:
    every: float = field(default=300.0, metadata=DUR)
    window: float = field(default=3600.0, metadata=DUR)


@dataclass
class Config:
    project: str | None = None
    bus: str = "file"
    metrics_store: str = "file"
    stores: list[Store] = field(default_factory=list)
    machines: list[Machine] = field(default_factory=list)
    items: list[Reducer] = field(default_factory=list)
    discovery: Discovery = field(default_factory=Discovery)

    def machine(self, name: str) -> Machine | None:
        return next((m for m in self.machines if m.name == name), None)

    def store(self, name: str) -> Store | None:
        return next((s for s in self.stores if s.name == name), None)
