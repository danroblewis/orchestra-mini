# Orchestra, minimal

Orchestra is an append-only **event log**, folded into **items** by subject, and a **metrics
store**, readable together with remote stores through **adapters**. It runs **machines** on a
schedule, on events, or continuously: a machine reads events as JSON lines on stdin and writes
events and metrics as JSON lines on stdout, and Orchestra feeds its stdin and routes its stdout.
From what it records, Orchestra knows who acts and discovers a **system dynamics model** from the
metrics. Everything a team does lives in its own scripts and prompts; Orchestra only runs them,
records what they say, and shows it. The six contracts are listed at the top of README.md; this
file states each exactly.

The owner, 2026-11-05: "This should be so damn simple that someone could read the whole config and
even edit it by hand." "Most of our interface is 'run a command' and 'parse the stdout'."

## 1. The config file

One file, YAML or JSON (the same schema), in layers merged in this order, the later winning:

1. `orchestra.yaml` (or `.yml`, `.json`) at the repository root, optional, shared by the team;
2. the private overlay, `~/.orchestra/<project>/orchestra.yaml` (or `.json`), the owner's own.

Maps merge key by key; `stores` and `machines` merge by `name`, an overlay entry replacing the
repository's of that name whole, and `{name: x, enabled: false}` removing it. Unknown keys are
refused with the key's path, and the fields of earlier versions (`kind`, `mode`, `prompt`,
`tools`, a store's `url`) are refused with what says them now. **Secrets are environment variables
only**: a literal secret in the file is refused. Durations are `30s`, `5m`, `2h`, `1d`.

```yaml
project: payments                  # optional: the name pages show (default: the repository's folder)
bus: file                          # default; or redis://… amqp://… kafka://…
metrics_store: file                # default; or http://host:8428 (VictoriaMetrics)

stores:                            # other stores, read in place through an adapter (section 4)
  - name: chronosphere
    command: orchestra-prometheus https://tenant.chronosphere.io/data/metrics --token-env CHRONOSPHERE_TOKEN
    match: '{__name__=~"payments_.*", env="prod"}'    # required: every query stays inside it
    kinds: {"*_inflight": level, "*_bucket": ignore}  # flow | level | ignore (default: by the name)
    names: {"payments_requests_total": "{service} requests"}   # renaming rules: glob -> template
    split: [service]               # one series per value of these labels
    per_day: 2000                  # requests a day at most (default 5000); rate: 1/s by default

machines:                          # what Orchestra runs (section 3)
  - name: github                   # pull requests, reviews and checks as events
    command: .orchestra/scripts/github_events.py
    every: 2m
  - name: ci_triage                # a red check: caused by the change, or flaky?
    command: orchestra decide "Is this failure caused by the change?" --answers real,flaky --inputs title,log_tail
    on: [github.check.failed]
  - name: fixer                    # a real failure: an agent pushes a fix to the pull request
    command: orchestra agent .orchestra/prompts/fixer.md --model sonnet --per-day-usd 15
    on: [ci_triage.real]
  - name: reviewer                 # one session that keeps its context across comments
    command: orchestra agent .orchestra/prompts/reviewer.md --per-day-usd 10
    on: [github.pr.review_requested, github.pr.commented]
    continuous: true
  - name: deploys                  # a program kept running: follows the deploy log
    command: .orchestra/scripts/deploy_tail.py
    continuous: true
    env: {DEPLOY_ENV: prod}
```

Also optional: `items: [{match: "task/*", command: …}]` (section 6) and `discovery: {every: 5m,
window: 60m}` (section 7).

## 2. Lines

A machine writes one JSON object per line on stdout. A line with `type` is an **event**, a line
with `metric` and `value` a **measurement**, a line with `cost_usd` a **run line**, and anything
else goes to the machine's log (as does its stderr).

```json
{"type": "github.pr.opened", "subject": "acme/api/pull/12", "source": "person.github.alice",
 "id": "gh:acme/api/pull/12:opened", "time": "2026-11-05T10:00:00Z", "data": {"title": "Fix retries"}}
{"metric": "queue_depth", "value": 12, "labels": {"queue": "triage"}, "kind": "gauge"}
{"cost_usd": 0.12, "note": "optional: a word about the run"}
```

An event: `type` (required, `<system>.<noun>.<verb>`); `subject`, the item it is about; `source`,
who acted (default the machine; people are `person.<system>.<account>`); `id`, stable per change
(an event whose id the bus has is dropped; default a hash of the line); `time` (RFC 3339, default
now); `kind`, what the subject is (default the type less its verb: `github.pr`); `data`, anything
(Orchestra reads it only for items). `{"type": "orchestra.actor", "source": …, "data": {"name": …}}`
names an actor for the pages.

A measurement: `metric` (required, named for what it measures), `value` (required, a number),
`labels`, `time` (RFC 3339 or Unix seconds; default now), `kind` (`counter` or `gauge`; default
`_total` and `_count` are counters). Every point is labelled `machine` and `project`; a series
named `orchestra_*` is refused. A once run whose first line is Prometheus text is read as
Prometheus text, written when the run succeeds.

A run line says what the run cost (`cost_usd`), with an optional `note` and `skipped: true` for a
run that did no work. Programs elsewhere may `POST` an event's JSON to the events endpoint, which
`orchestra run` prints with its token when it starts. The bus keeps every event in the CloudEvents
envelope (`specversion`, `id`, `source`, `type`, `time`, `subject`, `data`).

## 3. Machines

A machine has four fields: `name`, `command` (run without a shell, in the repository), when it
runs, and an optional `env` map added to its environment.

- **once** (the default): one run per event of the types in `on` (globs allowed), the event as one
  line on stdin and stdin then closed; with `every`, a run on that schedule with nothing on stdin
  (a tick is skipped while the last one runs). Both may be given. One run at a time, events waiting
  in order; a run is stopped after an hour.
- **`continuous: true`**: started when `orchestra run` starts and restarted when it exits; the
  events in `on` arrive on its open stdin as they come, one line each.

A machine never gets its own events. Every run gets `ORCHESTRA_MACHINE` and `ORCHESTRA_SOURCE`
(its name, what `orchestra emit` publishes as), `ORCHESTRA_EVENT` and `ORCHESTRA_ITEM` (the event
and its item's state; once), `ORCHESTRA_CONTINUOUS` (`1` when continuous), `ORCHESTRA_PROJECT`,
`ORCHESTRA_REPO` and `ORCHESTRA_RUN`, in the owner's environment, unstripped. Every run is
recorded (`orchestra query runs`): start, end, exit code, events, points, cost, note, log. A
continuous machine's process is one record, and each run line it prints ends a **turn**, a record
of its own for the oldest event it was given and not yet answered.

Two programs come with Orchestra, named in `command`; they read stdin and write lines like any:

- `orchestra agent <prompt file> [--model M] [--per-day-usd N]`: Claude Code (`claude -p`, with
  the owner's login, shell and credentials, `--dangerously-skip-permissions`) on the prompt file
  and each event with its item. Once: one turn per event. Continuous: one stream-json session,
  each event a user message, so it keeps its context. The JSON lines in what it says are its
  events and metrics; it may also run `orchestra emit '<json>'`. Each turn ends with a run line
  of its cost; past `--per-day-usd` spent today a turn is not started (`skipped`). What it may do
  is written in its prompt file, or in scripts beside it that it runs.
- `orchestra decide "<question>" --answers a,b,c [--inputs field,field] [--min-confidence 0.6]
  [--model M]`: one model call per event with the question, the event and the item's `inputs`
  fields (default all); it prints `decision.made` (`answer`, `confidence`, a one-line `reason`)
  and `<machine>.<answer>` (so `on: [ci_triage.real]` reads plainly), and a run line. Below
  `--min-confidence` the answer is `unsure`.

The Machines page shows every machine with its runs, and every `source` heard on the bus that
Orchestra doesn't run (people, other programs).

## 4. Stores

A store is read in place, never copied or written: its `command` is an **adapter**, a program
Orchestra keeps running and asks one request at a time. A store has `name`, `command`, `match`
(required; a match-everything selector is refused) and optionally `names`, `kinds`, `split`,
`rate` (default `1/s`), `per_day` (default 5000) and `env`.

Orchestra does everything around every adapter: each query is conjoined with `match` and needs a
selector of its own, reaches back a day at most and selects 500 series at most (counted with a
`series` request first); requests wait for `rate`, stop at `per_day` until 00:00 UTC, and are
logged (`<state>/stores/<name>.requests.jsonl`); an adapter that gives no answer in 60 s is killed,
and one that has exited is started again on the next request. `kinds` (`flow` read as a counter,
`level` as a gauge, `ignore`), `names` (`{label}` and `{metric}` in the template) and `split` are
applied to the answers. `orchestra-prometheus <url> [--token-env NAME] [--header NAME]` is the
adapter for any Prometheus HTTP API, a path prefix in the URL allowed.

### Writing a store adapter

An adapter reads one JSON request per line on stdin and answers each on stdout, in order, flushing
after each answer. Its stderr goes to `<state>/logs/stores/<name>.log`. It asks the store what it is
told; it never widens a query (Orchestra has already bounded it).

| request | answer lines |
|---|---|
| `{"op": "series", "match": "<selector>"}` | `{"metric": "<name>", "labels": {"<label>": "<value>", …}}`, one per series the selector selects |
| `{"op": "range", "query": "<selector or PromQL>", "start": <unix s>, "end": <unix s>, "step": <s>}` | `{"metric": "<name or empty>", "labels": {…}, "points": [[<unix s>, <number>], …]}`, one per result series |
| `{"op": "health"}` | `{"ok": true}`, or `{"ok": false, "error": "<why>"}` |

Every answer ends with the line `{"end": true}`. A request that fails is answered with one
`{"error": "<why, never a credential>"}` line, then `{"end": true}`. `labels` exclude the name;
`points` are in time order, numbers only (leave out NaN). The adapter takes its credentials from
environment variables named on its command line (`--token-env NAME`), never from the command line
itself, and keeps running until its stdin closes. Try it by hand:
`echo '{"op": "health"}' | <command>`.

## 5. Commands and pages

- `orchestra run [--web]`: makes the state directory (`~/.orchestra/<project>/`), loads and checks
  the config (refusing to start on an error: a config that doesn't load, a command not found, an
  `orchestra agent` prompt file missing; saying warnings: a store not answering), prints the events
  endpoint and its token, and runs every machine; `--web` also serves the pages and prints their
  login link.
- `orchestra chat [message] [--continue]`: the Supervisor, a Claude Code session in the repository
  briefed with this spec and what is set up. With no config yet, or one that doesn't load, it sets
  the project up (writes the config, its scripts and prompts, and proves them); otherwise it
  answers about Orchestra and what is happening, changes it, and clears the data when asked
  (`python -m orchestra.reset --yes`: the bus, metrics, items, runs, logs and model; never the
  config, tokens or scripts). `--continue` carries on the last session.
- `orchestra emit '<json>'`: one event or metric line, as `ORCHESTRA_SOURCE` (default `cli`).
- `orchestra query events [-n N] [--type T] [--subject S]`, `query items [--kind K] [--all]
  [show SUBJECT]`, `query metrics [prefix]`, `query model`, `query runs [machine]`, and
  `query '<promql>' [--store NAME]` (the metrics store, or a store through its adapter).

Pages, read only: **Dashboard** (what waits on the owner, open items, recent runs, spend),
**Items**, **Machines** (every machine and source, what each publishes and listens for, runs),
**Dynamics** (the model), **Core** (bus, metrics store, stores, machines: health and recent errors).

## 6. Items

An **item** is everything the bus says about one `subject`, folded into its current state: `kind`
(section 2); `fields`, every key of every event's `data`, the latest event (by `time`, ties by bus
order) winning, `null` removing a key; `type`, the latest event's; `first` and `last`, the times of
the first and latest events; `events`, its history; `done`, true once an event whose type ends
`.completed`, `.closed` or `.cancelled` arrived (a later event reopens it). A subject names one thing
unambiguously, the way its own system writes it (`acme/api/pull/3`, `acme/api/issues/1`, `ABC-123`).
A reducer replaces the fold for subjects matching a glob: `items: [{match: "task/*", command: …}]`
gets the item's events on stdin and prints its state as one JSON object. The Items page groups them
by kind and shows what each waits on (`fields.waiting_on`).

## 7. The model

Discovery runs every `discovery.every` (default 5m) over `discovery.window` (default 60m) of the
metrics: the store's own series, every store's series within its `match` (through its adapter), and
Orchestra's count of the bus, `orchestra_events_total{type, source}`. It finds stocks and flows,
fits their equations, saves a version when the structure changes, and guesses which machines move
each flow from timing. A **prediction** is an event `orchestra.prediction.made` (`data`: the stock
or flow, the expected direction or value, by when); Orchestra judges it when its time comes and
publishes `orchestra.prediction.judged`.

## 8. What is not here

No plugins, no tools section, no machine kinds, no declared model, no consent or confirmations, no
workflows engine, no planners, no outbox, no TOML. A team's way of working is which machine listens
for which events, in the config file, and the scripts and prompts it names. Anything more is added
only when a real flow needs it.
