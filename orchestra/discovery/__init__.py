"""Discovery (SPEC §6): the model guessed from the metrics, the seams ``orchestra run`` and
``orchestra model`` use (``orchestra.hooks``):

- ``start(project, stop)``: a thread that rediscovers every ``discovery.every`` over the last
  ``discovery.window``, writes the latest discovery, saves a version when the structure changed,
  and judges the predictions that are due (``predictions``); it returns at once;
- ``show(project, args) -> int``: ``orchestra model`` prints the latest discovery (``--json`` for
  its JSON, ``--versions`` for the versions, ``--now`` to discover first);
- ``run_once(project)``: one round, for both.

The strategy is ``conservation`` (``discover``); with the ``sd`` extra installed, ``kalman_lna``
(``strategies``) also explains what the counters leave unexplained."""

from __future__ import annotations

import json
import threading
import time
import traceback

from ..events import Events
from ..lines import Publisher
from . import predictions
from .discover import discover
from .model import Model, structure
from .series import gather
from .show import text
from .store import ModelStore
from .strategies import refine

STEP_PARTS = 60          # the window is read on a grid of this many steps (at least 10 s each)


def run_once(project, now: float | None = None) -> Model:
    cfg = project.config.discovery
    end = time.time() if now is None else now
    start = end - cfg.window
    step = max(10.0, cfg.window / STEP_PARTS)
    series, left, errors = gather(project, start, end, step)
    model = discover(series, start=start, end=end, step=step, name=project.name, left=left)
    model = refine(model, series, start=start, end=end, step=step)
    model.discovery["store_errors"] = errors
    store = ModelStore.for_project(project)
    current = store.load()
    if current is None or structure(current) != structure(model):
        model.discovery["version"] = store.save(model, "the structure changed" if current else "the first model")
    else:
        model.discovery["version"] = store.current_version()
    store.write_discovery(model, end)
    return model


def _loop(project, stop: threading.Event) -> None:
    events = Events.for_project(project)
    publisher = Publisher(events)
    log = project.path("logs", "discovery.log")
    while not stop.is_set():
        try:
            model = run_once(project)
            predictions.sync(project, events, publisher, model)
            note = f"{len(model.stocks)} stocks, {len(model.flows)} flows (version {model.discovery.get('version')})"
        except Exception:
            note = traceback.format_exc()
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {note}\n")
        stop.wait(project.config.discovery.every)


def start(project, stop: threading.Event) -> None:
    threading.Thread(target=_loop, args=(project, stop), daemon=True, name="discovery").start()


def show(project, args: list[str]) -> int:
    store = ModelStore.for_project(project)
    if "--versions" in args:
        for v in store.versions():
            print(f"v{v['version']}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(v['saved_at'] or 0))}  "
                  f"{v['stocks']} stocks, {v['flows']} flows  {v['note']}")
        return 0
    if "--now" in args:
        model, at = run_once(project), time.time()
    elif (got := store.latest_discovery()) is not None:
        at, model = got
    else:
        print("no model yet: orchestra run discovers it every discovery.every, or orchestra model --now")
        return 1
    print(json.dumps(model.to_json(), indent=1) if "--json" in args else text(model, at))
    return 0
