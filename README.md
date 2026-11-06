# Orchestra

Orchestra is an append-only **event log**, folded into **items** by subject, and a **metrics
store**, readable together with remote stores through **adapters**. It runs **machines** on a
schedule, on events, or continuously: a machine reads events as JSON lines on stdin and writes
events and metrics as JSON lines on stdout, and Orchestra feeds its stdin and routes its stdout.
From what it records, Orchestra knows who acts (every source on the log) and discovers a **system
dynamics model** (from the metrics).

## The six contracts

1. **An event line**: `{"type", "subject", "source", "id", "time", "kind", "data"}`; only `type` is
   required. Orchestra fills the rest and drops an id it has seen.
2. **A metric line**: `{"metric", "labels", "value", "time"}` (Prometheus text is accepted too).
3. **The fold**: an item is every event with one `subject`. Its `kind` comes from the first event
   (or its type less the verb), its `fields` are the latest value of each `data` key by event time
   (`null` removes one), and it is done when its latest type ends `.completed`, `.closed` or
   `.cancelled`. A reducer program can replace the fold for chosen subjects.
4. **A machine**: `name`, `command`, and when it runs: `on:` event types, `every:`, or
   `continuous: true`. A once run gets its event on stdin, then stdin closes; a continuous one keeps
   stdin open. Its name, the event and the item's state are in its environment. Agents and decisions
   are machines too, through the built-in programs `orchestra agent` and `orchestra decide`.
5. **A store adapter**: a running program answering three requests as JSON lines (`series`,
   `range`, `health`); Orchestra guards, caps and renames around it. `orchestra-prometheus` ships
   for Prometheus-compatible stores.
6. **What Orchestra derives**, never written by users: the machines and people that act (from the
   events' sources), the model (from the metrics, with its guess at who moves each flow), and the
   predictions judged against the metrics.

People use one config file (machines and stores), four commands (`run`, `chat`, `emit`, `query`) and
read-only pages. The details are [SPEC.md](SPEC.md).

## Install

```bash
gh auth setup-git      # the repository is private
uv tool install --force 'orchestra @ git+https://github.com/danroblewis/orchestra-mini@main'
# with the Kalman discovery extra:
uv tool install --force 'orchestra[sd] @ git+https://github.com/danroblewis/orchestra-mini@main'
```

## Use

```bash
cd <your repository>
orchestra chat         # the Supervisor: with no config it sets the project up (or: orchestra chat "watch our Jira");
                       # then it answers about what is happening, changes things, clears the data; --continue resumes
orchestra run --web    # checks the config, runs every machine, serves the pages; Ctrl-C stops it
orchestra emit '{"type": "deploy.release.done", "subject": "v1.4"}'
orchestra query items  # also: events, metrics, model, runs, or '<promql>' [--store NAME]
```

A machine is four fields, so the config stays readable by hand:

```yaml
machines:
  - name: ci_triage
    command: orchestra decide "Is this failure caused by the change?" --answers real,flaky
    on: [github.check.failed]
  - name: fixer
    command: orchestra agent .orchestra/prompts/fixer.md --per-day-usd 15
    on: [ci_triage.real]
```

`orchestra agent` and `orchestra decide` are programs that come with Orchestra; any other command
works the same way. An agent runs Claude Code with your own login and environment.

## Develop

```bash
uv sync --extra dev             # --extra sd for the Kalman discovery
uv run python -m pytest -q
uv run ruff check
```
