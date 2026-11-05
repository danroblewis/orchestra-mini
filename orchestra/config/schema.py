"""The config's schema (SPEC §1): plain dataclasses, one per section.

A field's metadata says how it is read: ``required``; ``duration`` (a string like ``30s``, ``5m``,
``2h``, ``1d``, or seconds, kept as seconds); ``names`` (a string or a list, kept as a list).
Every other field has its default. ``load`` builds these from the merged layers."""

from __future__ import annotations

from dataclasses import dataclass, field

REQ = {"required": True}
DUR = {"duration": True}
LIST = {"names": True}
KINDS = ("agent", "decision", "program")


@dataclass
class Auth:
    header: str = "Authorization"
    token_env: str | None = None


@dataclass
class Store:
    name: str = field(default="", metadata=REQ)
    url: str = field(default="", metadata=REQ)
    match: str = field(default="", metadata=REQ)
    token_env: str | None = None
    auth: Auth | None = None
    kinds: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    split: list[str] = field(default_factory=list, metadata=LIST)
    rate: str = "1/s"
    per_day: int = 5000


@dataclass
class MetricsCommand:
    name: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)
    every: float = field(default=60.0, metadata=DUR)
    timeout: float = field(default=30.0, metadata=DUR)


@dataclass
class EventsCommand:
    name: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)
    every: float = field(default=60.0, metadata=DUR)
    timeout: float = field(default=30.0, metadata=DUR)
    continuous: bool = False


@dataclass
class Tool:
    name: str = field(default="", metadata=REQ)
    command: str = field(default="", metadata=REQ)
    help: str | None = None


@dataclass
class Process:
    name: str = field(default="", metadata=REQ)
    kind: str = "agent"
    prompt: str | None = None            # agent: the instructions file
    command: str | None = None           # program: what to run
    question: str | None = None          # decision
    answers: list[str] = field(default_factory=list, metadata=LIST)
    inputs: list[str] = field(default_factory=list, metadata=LIST)
    min_confidence: float = 0.6
    on: list[str] = field(default_factory=list, metadata=LIST)
    tools: list[str] | None = field(default=None, metadata=LIST)   # None: every tool
    model: str | None = None
    every: float | None = field(default=None, metadata=DUR)
    continuous: bool = False
    timeout: float = field(default=1200.0, metadata=DUR)
    per_day_usd: float = 10.0


@dataclass
class Agent:
    harness: str = "claude_code"
    command: str = "claude"


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
    metrics: list[MetricsCommand] = field(default_factory=list)
    events: list[EventsCommand] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    processes: list[Process] = field(default_factory=list)
    agent: Agent = field(default_factory=Agent)
    items: list[Reducer] = field(default_factory=list)
    discovery: Discovery = field(default_factory=Discovery)

    def tool(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)

    def process(self, name: str) -> Process | None:
        return next((p for p in self.processes if p.name == name), None)
