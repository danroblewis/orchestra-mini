"""A fake store adapter (SPEC §4's protocol), run as a program: it answers ``series`` and ``range``
from the points in the JSON file ``FAKE_ADAPTER_DATA`` with Orchestra's PromQL subset, and
``health`` with ok. It appends every request to ``FAKE_ADAPTER_LOG``. A query naming ``boom``
answers an error line; one naming ``hang`` never answers; and when the file ``FAKE_ADAPTER_DIE``
exists it is removed and the adapter exits without answering (a dead adapter)."""

import json
import os
import sys
import time
from pathlib import Path

from orchestra.metrics import promql


def main():
    dump = [(n, lab, [tuple(p) for p in pts]) for n, lab, pts in json.loads(Path(os.environ["FAKE_ADAPTER_DATA"]).read_text())]
    for raw in sys.stdin:
        req = json.loads(raw)
        with open(os.environ["FAKE_ADAPTER_LOG"], "a") as f:
            f.write(json.dumps(req) + "\n")
        die = os.environ.get("FAKE_ADAPTER_DIE")
        if die and os.path.exists(die):
            os.remove(die)
            sys.exit(3)
        q = str(req.get("query") or req.get("match") or "")
        if "hang" in q:
            time.sleep(3600)
        out = []
        if "boom" in q:
            out = [{"error": "the store said boom"}]
        elif req["op"] == "health":
            out = [{"ok": True}]
        elif req["op"] == "series":
            seen = {}
            for lab, _ in promql.select(dump, q, 0, time.time() + 60):
                seen.setdefault(tuple(sorted(lab.items())), lab)
            out = [{"metric": lab.get("__name__", ""), "labels": {k: v for k, v in lab.items() if k != "__name__"}} for lab in seen.values()]
        elif req["op"] == "range":
            for lab, pts in promql.query_range(dump, q, float(req["start"]), float(req["end"]), float(req["step"])):
                lab = dict(lab)
                out.append({"metric": lab.pop("__name__", ""), "labels": lab, "points": [[t, v] for t, v in pts]})
        print("the adapter's chatter, not a JSON line", flush=True)
        for x in out + [{"end": True}]:
            print(json.dumps(x), flush=True)


if __name__ == "__main__":
    main()
