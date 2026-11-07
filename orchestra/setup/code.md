# How the code looks

This code is the project's own: the team will read it, change it, and may move it into their
repository's code. Write it the way a careful engineer would write a small, well-factored tool.

- **One folder, one kit.** Put every collector, reactor and projection in `.orchestra/` (prompts in
  `.orchestra/prompts/`), beside the project's own kit, `.orchestra/kit.py`, which you write. It
  is short and has no dependencies, and every script starts `from kit import …`. The kit holds:
  - `Fact`, `Metric` and `View` as dataclasses (a `View` has its `subject`, or `key`, and its
    fields), and a `Fact` read from stdin is one too, with `data` as given;
  - `facts()`, the facts on stdin as `Fact`s; `views()`, the run's views file as a dict;
  - `run(things)`, which writes each `Fact`, `Metric` or `View` it is given as one JSON line, so no
    script calls `print` or `json.dumps`.
- **Each script is one generator function and a call to `run`.** A collector is
  `run(collect())`, a reactor `run(react(fact, views()))` for each fact from `facts()`, a projection
  `run(project(facts()))`; each function yields `Fact`s, `Metric`s or `View`s. Keep the reading of a source (HTTP, paging) in a small module of its own
  (`.orchestra/jira_api.py`), shared by the scripts that need it.
- **Name things.** Turn a source's payload into small named records or functions once
  (`Review.from_api(raw)`, `issue_status(issue)`), and use those, not chains of `x["a"]["b"]` and
  nested comprehensions. Short functions with plain loops read better than clever one-liners.

```python
#!/usr/bin/env python3
"""Each branch's CI status, observed (CI keeps no history)."""
from ci_api import branches
from kit import Fact, Metric, run

REPO = "acme/api"


def collect():
    for b in branches():
        yield Fact(type="ci.branch.observed", subject=f"{REPO}/branch/{b.name}",
                   data={"branch": b.name, "status": b.status})
        yield Metric("ci_branch_failing", int(b.status == "failing"), labels={"branch": b.name})


run(collect())
```
