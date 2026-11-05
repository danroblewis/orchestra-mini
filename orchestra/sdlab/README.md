# SD Lab's kalman_lna, inside Orchestra

SD Lab's discovery research, as it was benchmarked: the `kalman_lna` strategy (stocks as hidden
states moved by flows, counting noise as process noise, the reported levels as observations of
them) and the modules it imports, nothing else. `identify(stocks, run, opts)` finds a stock's flows
from its levels; `opts["known"]` is what is already known about each step (the counters' changes),
so it explains only the rest.

- **Source:** the old Orchestra's `orchestra/sdlab` (SD Lab commit `7edb72d`, 2026-10-24), itself
  `sdlab/discovery/` with imports rewritten to `orchestra.sdlab.`. Here only the import closure of
  `discovery/strategies/kalman_lna` and `discovery/api.py` (88 modules of the ~50 strategies' 340,000
  lines), copied unchanged.
- **Dependencies:** the `sd` extra (`uv sync --extra sd`: numpy, scipy, numba, highspy).
- **How Orchestra uses it:** `orchestra/discovery/strategies.py`.

Lint (ruff) skips this directory: it is research code kept as it was benchmarked.
