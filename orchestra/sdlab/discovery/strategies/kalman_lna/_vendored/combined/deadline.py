# Vendored for kalman_lna from sdlab/discovery/strategies/combined/deadline.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A time cap on one specialist, nested inside whatever cap the caller already set.

The ladder's runner limits each item with SIGALRM. A specialist that would use the whole item's time
(the lna family on a realistic model) is stopped earlier here, so the others still run: the outer
timer is read, a shorter one set, and the outer one put back afterwards with the time used taken
off. `Overrun` derives from BaseException, so a specialist's own `except Exception` does not
swallow it. Off the main thread, or without SIGALRM, nothing is capped.
"""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager


class Overrun(BaseException):
    pass


def outer_left() -> float | None:
    """Seconds left on the caller's timer (None when there is none)."""
    if not hasattr(signal, "getitimer"):
        return None
    left = signal.getitimer(signal.ITIMER_REAL)[0]
    return left if left > 0 else None


@contextmanager
def deadline(seconds: float | None):
    usable = seconds and seconds > 0 and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return
    left = outer_left()
    if left is not None and left <= seconds:
        yield                                              # the caller's own timer comes first
        return
    began = time.perf_counter()
    prev = signal.getsignal(signal.SIGALRM)

    def fire(*_):
        raise Overrun(f"over {seconds:.0f} s")
    signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)
        if left is not None:
            signal.setitimer(signal.ITIMER_REAL, max(1e-3, left - (time.perf_counter() - began)))
