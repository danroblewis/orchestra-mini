# Event-bus conformance suite

One suite for every `orchestra.bus` backend. The file backend always runs; each server backend
runs when its URL is set.

```
docker compose -f deploy/dev/compose.yaml up -d        # Redis, RabbitMQ, Kafka, VictoriaMetrics
uv sync --extra dev --extra bus
ORCHESTRA_TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
ORCHESTRA_TEST_AMQP_URL=amqp://orchestra:orchestra@127.0.0.1:5672/ \
ORCHESTRA_TEST_KAFKA_URL=kafka://127.0.0.1:9092 \
uv run python -m pytest tests/bus -q
```

Use `127.0.0.1`, not `localhost`: the containers listen on IPv4 only, and Kafka clients try IPv6
first. Every test uses its own stream and removes it afterwards.

What the suite holds every backend to:
- order;
- cursors, including the loop's startup idiom `read(0, limit=1)`;
- a cursor past the end;
- a second instance on the same stream;
- concurrent publishers, both threads and separate processes, with nothing lost and each
  publisher's order kept;
- history surviving a reopen;
- 200 KB and unicode payloads;
- isolation between projects;
- the `Events` facade;
- configuration choosing the backend.
