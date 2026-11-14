# Orchestra

Orchestra is an append-only **log of facts**, read through **projections** into **views**, and a
**metrics store**, readable together with remote stores through **adapters**. It runs **machines**
on a schedule, on facts, or continuously: a machine is a fact-in, fact-out stream processor that
reads facts as JSON lines on stdin and writes facts and metrics as JSON lines on stdout, and
Orchestra feeds its stdin and routes its stdout. An event is one kind of fact, the commonest; the
fact is the base. From what it records, Orchestra knows who acts (every source on the log) and
discovers a **system dynamics model** (from the metrics).

## The six contracts

1. **A fact line**: `{"type", "subject", "source", "id", "time", "kind", "data", "causedby"}`; only
   `type` is required. Orchestra fills the rest, drops an id it has seen, skips a repeated
   observation, and stamps the envelope with `causedby`, `depth` and `recordedtime`.
2. **A metric line**: `{"metric", "labels", "value", "time"}` (Prometheus text is accepted too).
3. **A projection**: `name`, `match` (item globs), `command`, and optionally `types` and
   `whole: true`. Its command reads the facts of the items being built on stdin, in log order,
   and prints one view per item (`{"subject": …}`), or per key it chooses when whole
   (`{"key": …}`). Views are rebuilt, never logged. The built-in `latest` covers every item:
   the latest value of each `data` key by time, the latest `data` of each type, and `done`.
4. **A machine**: `name`, `command`, and when it runs: `on:` fact types, `every:`, or
   `continuous: true`. A once run gets its fact on stdin, then stdin closes, and its item's
   views in the file `ORCHESTRA_VIEWS_FILE`; a continuous one keeps stdin open. What it writes is
   stamped with the fact that caused it. Agents and decisions are machines too, through the
   built-in programs `orchestra agent` and `orchestra decide`.
5. **A store adapter**: a running program answering three requests as JSON lines (`series`,
   `range`, `health`); Orchestra guards, caps and renames around it. `orchestra-prometheus` ships
   for Prometheus-compatible stores.
6. **What Orchestra derives**, never written by users: the machines and people that act (from the
   facts' sources), the model (from the metrics, with its guess at who moves each flow), and the
   predictions judged against the metrics.

People use one config file (machines, projections and stores), four commands (`run`, `chat`, `emit`, `query`) and
read-only pages, but for one box: on the Dashboard, type what you want done and the built-in `worker`
(an agent) does it in the repository and commits it. That needs no config. The details are [SPEC.md](SPEC.md).

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
orchestra query views  # also: facts, check, metrics, model, runs, or '<promql>' [--store NAME]
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
