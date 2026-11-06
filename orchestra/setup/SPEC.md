# Orchestra, minimal

Orchestra is a dumb system. It keeps an **event bus** and a **metrics store**, runs **machines**
(stream processors: events in as JSON lines on stdin, events and metrics out as JSON lines on
stdout), folds events into **items**, knows the other machines that act from the bus, and guesses
a **system dynamics model** from the metrics. Everything a team does lives in its own scripts and
prompts; Orchestra only runs them, records what they say, and shows it.

The owner, 2026-11-05: "This should be so damn simple that someone could read the whole config and
even edit it by hand." "Most of our interface is 'run a command' and 'parse the stdout'." And:
every processing entity is a machine, the collectors and the AI agents too.

## 1. The config file

One file, YAML or JSON (the same schema), in layers merged in this order, the later winning:

1. `orchestra.yaml` (or `.yml`, `.json`) at the repository root, optional, shared by the team;
2. the private overlay, `~/.orchestra/<project>/orchestra.yaml` (or `.json`), the owner's own.

Maps merge key by key; a list of named entries (`stores`, `tools`, `machines`) merges by `name`, an overlay entry replacing the repository's of that name whole, and
`{name: x, enabled: false}` removing it. Unknown keys are refused with the key's path, so a typo
never silently does nothing. **Secrets are environment variables only**: a field ending `_env` names
one (`token_env: JIRA_TOKEN`); a literal secret in the file is refused. Durations are `30s`, `5m`,
`2h`, `1d`. Every field not marked required has a default.

```yaml
project: payments              # optional: the name pages show (default: the repository's folder)

bus: file                      # default file (the state directory); or redis://… amqp://… kafka://…
metrics_store: file            # default file; or http://host:8428 (VictoriaMetrics)

stores:                        # other stores, read in place, never copied (Prometheus API)
  - name: chronosphere         # required
    url: https://tenant.example.io/data/metrics   # required
    token_env: CHRONOSPHERE_TOKEN                 # bearer; or auth: {header: X-Key, token_env: …}
    match: '{__name__=~"queue_.*", env="prod"}'   # required; a match-everything selector is refused
    kinds: {"*_total": flow, "*_inflight": level, "*_bucket": ignore}
    names: {"grpc_server_*": "{service} {metric}"}   # renaming rules: glob -> template
    split: [service]           # one series per value of these labels
    rate: 1/s                  # default 1/s; requests per second at most
    per_day: 5000              # default 5000 requests a day

tools:                         # commands agents may call (section 4)
  - name: notify_slack
    command: scripts/slack_post.py {channel} {text}
    help: Post a message to a Slack channel.   # optional when the command answers --help
  - name: open_pr
    command: gh pr create --draft --head {branch} --title {title} --body-file {body_file}

machines:                      # what Orchestra runs (section 3)
  - name: jira_counts          # required; a collector: a program on a schedule
    command: npm run get_metrics   # program: required; run without a shell, in the repository
    every: 5m                  # once, scheduled: run every 5m, nothing on stdin
  - name: jira
    command: scripts/jira_events.py
    mode: continuous           # once (default) | continuous: kept running, stdin open
  - name: worker
    kind: agent                # program (default) | agent | decision
    prompt: prompts/worker.md  # agent: the instructions; each event and its item are added
    on: [task.researched, review.changes, pr.commented]   # one run per event (continuous: fed each)
    tools: [open_pr]           # default: every tool
    model: sonnet              # agent and decision: default the harness's own
    timeout: 30m               # once: default 20m
    per_day_usd: 10            # agent and decision: spend cap, default 10

agent:                         # how an agent machine runs: default Claude Code
  harness: claude_code         # claude_code | pi | codex
  command: claude              # the binary
```

## 2. Lines

A machine writes one JSON object per line on stdout. A line with `metric` and `value` is a
**measurement**, a line with `type` an **event**, and anything else goes to the machine's log (as
does its stderr).

```json
{"type": "jira.issue.transitioned", "subject": "ABC-123", "source": "person.jira.alice",
 "id": "jira:ABC-123:10042", "time": "2026-11-05T10:00:00Z", "data": {"from": "To Do", "to": "In Progress"}}
{"metric": "queue_depth", "value": 12, "labels": {"queue": "triage"}, "kind": "gauge"}
```

An event:

- `type` (required): `<system>.<noun>.<verb>`.
- `subject`: the item it is about (a ticket key, `owner/repo#12`, `task/abc`). Events without one are kept, about no item.
- `source`: who acted; default the machine's name. People are `person.<system>.<account>`.
- `id`: stable per change; Orchestra drops an event whose id it has. Default: a hash of the line.
- `time`: when it happened (RFC 3339); default now.
- `kind`: what the subject is (`github.pr`, `github.issue`, `jira.issue`, `task`); default: the
  first event's type less its verb (`github.pr.opened` → `github.pr`).
- `data`: anything; Orchestra never reads it, except for items (section 5).
- A line `{"type": "orchestra.actor", "source": "person.jira.alice", "data": {"name": "Alice A"}}`
  names an actor for the pages.

A measurement:

- `metric` (required): the series' name, for what it measures (`queue_depth`, `reviews_done_total`).
- `value` (required): a number.
- `labels`: a map of label names to values.
- `time`: RFC 3339 or Unix seconds; default when it was read.
- `kind`: `counter` (a running total, read by its increments) or `gauge` (a level); default by the
  name: `_total` and `_count` are counters, the rest gauges.

Every point is labelled `machine=<name>` (a label of that name on the line is kept as
`exported_machine`) and `project`. Series named `orchestra_*` are Orchestra's own and refused.
**Prometheus text** works too: a once run whose first non-empty line is a Prometheus sample or a
`#` line is read as Prometheus text (`name{label="v"} value [ms]`), written when the run succeeds.

Programs elsewhere may also `POST` an event's JSON to the events endpoint (`orchestra endpoint`
prints it). The bus records every event with the CloudEvents envelope (`specversion`, `id`,
`source`, `type`, `time`, `subject`, `data`); Orchestra fills what the line leaves out.

## 3. Machines

A **machine** is a stream processor: it reads events as JSON lines on stdin and writes events and
metrics as JSON lines on stdout (section 2). A collector is the case that ignores stdin. Each
machine has a kind:

- **program**: `command`, run without a shell, in the repository.
- **agent**: an AI agent (Claude Code `claude -p` by default) on the prompt file, with its tools and
  how to say what happened appended, and each event with the item's state; it acts with its own
  shell and the owner's credentials (the environment Orchestra runs in, unstripped). Its lines are
  those in what it says, or those it writes with `orchestra emit '<json>'` from its shell.
- **decision**: one model call with `question` and `answers` (and `inputs`: the item fields to
  include); it publishes `decision.made` with the answer, its confidence and a one-line reason, and an
  event `<machine>.<answer>` (so `on: ci_triage.real` reads plainly). Below `min_confidence`
  (default 0.6) the answer is `unsure`. Always once.

And a mode:

- **once** (the default): started for one event, which is written to stdin, then stdin is closed.
  It may work for minutes and write lines as it goes. With `on`, one run per event of those types (a
  glob may name them; events arriving during a run wait in order); with `every`, a run on that
  schedule with nothing on stdin (a tick whose last run hasn't finished is skipped). Both may be given.
- **continuous**: started when `orchestra run` starts and restarted when it exits; stdin stays open
  and the events in `on` arrive as they come, one line each. A continuous agent is
  `claude -p --input-format stream-json --output-format stream-json`: each event is one user
  message (`{"type": "user", "message": {"role": "user", "content": "<the event and the item>"}}`),
  its prompt is in its system prompt, and each turn's result gives that turn's cost.

A machine never gets its own events. Every run gets `ORCHESTRA_MACHINE` and `ORCHESTRA_SOURCE` (its
name: what `orchestra emit` and `orchestra tool` publish as), `ORCHESTRA_EVENT` (the event, once),
`ORCHESTRA_ITEM` (the item's state), `ORCHESTRA_PROJECT`, `ORCHESTRA_REPO` and `ORCHESTRA_RUN`.
Every run is recorded (`orchestra runs`): start, end, exit code, events published, points
written, cost when the harness reports it, its log; a continuous program's run is its process's
life, a continuous agent's run one turn. An agent or a decision past its `per_day_usd` today is
not started (a continuous agent is given no events) until the next day, and its run record says so.

The Machines page shows every configured machine with its runs and what it published, and every
`source` heard on the bus that Orchestra doesn't run (the people a collector reports, other programs).

## 4. Tools

A tool is a command an agent may call. `{name}` placeholders are filled from the call's arguments,
each as one argument (never through a shell). An agent's prompt lists its tools with their `help`,
or, when `help` is absent, the first 20 lines of `command --help`. Agents call a tool by running
`orchestra tool <name> key=value …`, which runs the command, prints its stdout, and publishes any
JSON event lines in it as the calling machine. A tool is how the config says what an agent may do
beyond reading: writing to Slack or opening a PR is in the file because the owner put it there.

## 5. Items

An **item** is everything the bus says about one `subject`, folded into its current state. Each
item has a **kind**, what it is (section 2's `kind`). Things of different kinds never share a
subject: a subject names one thing unambiguously, best the way its own system writes it
(`owner/repo/pull/3`, `owner/repo/issues/1`, `ABC-123`), even where two kinds share a numbering.

- `kind`: the item's kind;
- `fields`: every key of every event's `data`, the latest event (by `time`, ties by bus order)
  winning; `data` set to `null` removes a key;
- `type`: the type of the latest event; `first`, `last`: the times of the first and latest events;
- `events`: the item's history, in time order;
- `done`: true once an event whose type ends `.completed`, `.closed` or `.cancelled` arrived
  (an event after it reopens it).

An optional reducer replaces the fold for subjects matching a pattern: `items: [{match: "task/*",
command: scripts/task_state.py}]` gets the item's events as JSON lines on stdin and prints its state
as one JSON object. `orchestra items [--kind K]` lists open items (newest first, paged); `orchestra items
show <subject>` prints one with its history. The Items page filters by kind, with each kind's count. The Items page shows them, closed ones hidden unless
asked for, with what each waits on when its state says (`fields.waiting_on`).

## 6. The model

Discovery runs every `discovery.every` (default 5m) over `discovery.window` (default 60m) of the
metrics: the store's own series, every store's series within its `match` (with its kinds, names and
split), and Orchestra's count of the bus, `orchestra_events_total{type, source}`. It finds stocks
and flows and fits their equations, saves a version when the structure changes, and guesses which
machines move each flow from timing. A **prediction** is an event `orchestra.prediction.made`
(`data`: the stock or flow, the expected direction or value, by when); Orchestra judges it when its
time comes and publishes `orchestra.prediction.judged`.

## 7. Commands and pages

- `orchestra setup [what you want]`: a Claude Code session briefed with this spec and what is already
  set up (the config, `orchestra check`, credential variable names, `gh`'s login), which designs and
  writes the config, its scripts and prompts, proves them against the real systems, and reports.
- `orchestra chat [message] [--continue]`: the Supervisor, a Claude Code session briefed with this
  spec, the setup and what is happening now (open items, runs, events), which answers about
  Orchestra and changes it; `--continue` carries on the last session (`setup` takes it too).
- `orchestra reset [--yes]`: clears the data (bus, metrics, items, runs, logs, model) and keeps the
  config and tokens; refused while `orchestra run` runs.
- `orchestra init`: makes the state directory and, if asked, an empty `orchestra.yaml`.
- `orchestra run [--web]`: runs every machine; `--web` also serves the pages.
- `orchestra check`: validates the config and says each store's, tool's and machine's health.
- `orchestra events [--type T] [--subject S] [-n N]`.
- `orchestra emit '<json>'`: one event or one metric line, as `ORCHESTRA_SOURCE` (default `cli`)
  unless the line names its source.
- `orchestra metrics [prefix]`, `orchestra query '<promql>' [--store NAME]`.
- `orchestra items [show S]`, `orchestra runs [machine]`, `orchestra model`, `orchestra tool NAME …`.

Pages, read only: **Dashboard** (what waits on the owner, open items, recent runs), **Items**,
**Machines** (every machine and every source heard, what it publishes and listens for, its runs),
**Dynamics** (the model), **Core** (bus, metrics store, stores, machines: each one's health and
recent errors).

## 8. What is not here

No plugins, no declared model, no consent or confirmations, no workflows engine, no offers or
leases, no planners, no outbox, no bridges kits, no people map, no TOML. A team's way of working is
which machine listens for which events, in the config file, and the scripts and prompts it names.
Anything more is added only when a real flow needs it.
