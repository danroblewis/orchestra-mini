"""Projections and their views (SPEC §6): programs that read facts and print views, built by one
builder (``builder``), run one at a time (``run``), read with ``behind`` (``read``), and the views a
run saw (``seen``)."""

from .builder import Builder, projections
from .read import Views, said

__all__ = ["Builder", "Views", "projections", "said"]
