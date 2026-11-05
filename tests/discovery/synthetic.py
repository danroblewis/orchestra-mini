"""A synthetic metrics history: one queue (a stock), arrivals into it at a constant rate and
service out of it proportional to its length, a worker whose events on the bus are the services,
another source whose events are unrelated, and a gauge that is no stock."""

from __future__ import annotations

import random

from orchestra.metrics import Point

ARRIVALS = 2.0      # per minute
SERVICE = 0.25      # per minute per item waiting


def history(end: float, minutes: int = 60, seed: int = 7) -> list[Point]:
    rnd = random.Random(seed)
    t, start = end - minutes * 60, end - minutes * 60
    queue, arrived, served, noise = 5, 0, 0, 0
    out = [Point("queue_depth", queue, start, {"queue": "triage"}), Point("arrivals_total", 0, start, {}),
           Point("served_total", 0, start, {})]
    while True:
        rate_in, rate_out = ARRIVALS / 60, SERVICE * queue / 60
        t += rnd.expovariate(rate_in + rate_out)
        if t >= end:
            break
        if rnd.random() < rate_in / (rate_in + rate_out):
            queue, arrived = queue + 1, arrived + 1
            out.append(Point("arrivals_total", arrived, t, {}))
        else:
            queue, served = queue - 1, served + 1
            out.append(Point("served_total", served, t, {}))
            out.append(Point("orchestra_events_total", served, t, {"type": "worker.job.done", "source": "worker"}))
        out.append(Point("queue_depth", queue, t, {"queue": "triage"}))
    for i in range(minutes):
        out.append(Point("room_temperature", 20 + rnd.random(), start + 60 * i, {}))
        if rnd.random() < 0.3:
            noise += 1
            out.append(Point("orchestra_events_total", noise, start + 60 * i + 17, {"type": "chat.message.posted", "source": "chatter"}))
    return sorted(out, key=lambda p: p.ts)
