"""A fake ``claude`` for the agent and decision kinds: it saves its arguments and prompt, and
answers in the CLI's formats. With ``--json-schema`` it answers a decision (``FAKE_ANSWER``,
``FAKE_CONFIDENCE``); otherwise it streams an agent turn whose text says one event, prints
another outside the stream, calls ``orchestra tool FAKE_TOOL_CALL`` when set, and ends with a
result costing ``FAKE_COST``."""

CLAUDE = r'''
import json, os, sys, pathlib, shlex, subprocess
prompt = sys.stdin.read()
out = pathlib.Path(os.environ.get("FAKE_DIR", "."))
n = len(list(out.glob("call-*.json")))
(out / f"call-{n}.json").write_text(json.dumps({"argv": sys.argv[1:], "prompt": prompt,
    "event": os.environ.get("ORCHESTRA_EVENT"), "item": os.environ.get("ORCHESTRA_ITEM"),
    "process": os.environ.get("ORCHESTRA_PROCESS")}))
cost = float(os.environ.get("FAKE_COST", "0.25"))
if "--json-schema" in sys.argv:
    answer = {"answer": os.environ.get("FAKE_ANSWER", "real"), "confidence": float(os.environ.get("FAKE_CONFIDENCE", "0.9")),
              "reason": "it looks real"}
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": json.dumps(answer),
                      "structured_output": answer, "total_cost_usd": cost}))
    sys.exit(0)
subject = json.loads(os.environ["ORCHESTRA_EVENT"]).get("subject") if os.environ.get("ORCHESTRA_EVENT") else None
said = json.dumps({"type": "agent.work.done", "subject": subject, "data": {"ok": True}})
print(json.dumps({"type": "system", "subtype": "init", "session_id": "s"}))
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Working on it.\n" + said}]}}))
print(json.dumps({"type": "agent.note.left", "subject": subject}))
if os.environ.get("FAKE_TOOL_CALL"):              # the agent calls a tool from its shell
    subprocess.run(["orchestra", "tool", *shlex.split(os.environ["FAKE_TOOL_CALL"])], check=True, stdout=sys.stderr)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "Done.\n" + said,
                  "total_cost_usd": cost}))
'''
