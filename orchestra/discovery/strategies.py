"""The strategies: ``conservation`` (``discover``) always; ``kalman_lna``, SD Lab's, when the ``sd``
extra is installed (numpy, scipy). Placeholder until the extra is brought over."""

from __future__ import annotations


def installed() -> bool:
    return False


def refine(model, series, *, start: float, end: float, step: float):
    return model
