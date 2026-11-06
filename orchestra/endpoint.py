"""The events endpoint (SPEC §2): ``POST /events`` with one event as JSON (the same as a stdout
line), behind the project's token (``Authorization: Bearer <token>``), published once like any
other line (source default ``endpoint``); ``GET /health`` without the token. ``orchestra run``
serves it on 127.0.0.1 at a port the system picks and records its address in
``<state>/endpoint.json``, and prints it and the token when it starts. The token is kept in
``<state>/events.token``, readable by its owner only."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .lines import event

MAX_BODY = 1_000_000


def token(project) -> str:
    path = project.path("events.token")
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    return path.read_text().strip()


def recorded(project) -> dict | None:
    try:
        return json.loads((project.state_dir / "endpoint.json").read_text())
    except (FileNotFoundError, ValueError):
        return None


def handler(publisher, secret: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/events":
                return self._send(404, {"error": "not found"})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {secret}"):
                return self._send(401, {"error": "Authorization: Bearer <token> (orchestra run prints it)"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": f"at most {MAX_BODY} bytes"})
            body = self.rfile.read(n)
            try:
                raw = json.loads(body)
            except ValueError:
                return self._send(400, {"error": "not JSON"})
            ev, why = event(raw, "endpoint", hashlib.sha256(body).hexdigest()[:32]) if isinstance(raw, dict) else (None, "not an object")
            if ev is None:
                return self._send(400, {"error": why})
            fresh = publisher.publish(ev)
            self._send(201 if fresh else 200, {"id": ev["id"], "duplicate": not fresh})

        def log_message(self, *a):
            pass
    return Handler


def serve(project, publisher, stop: threading.Event, host: str = "127.0.0.1", port: int = 0) -> str:
    """Serve until ``stop``; the URL it listens on."""
    server = ThreadingHTTPServer((host, port), handler(publisher, token(project)))
    url = f"http://{host}:{server.server_address[1]}"
    project.path("endpoint.json").write_text(json.dumps({"url": url, "pid": os.getpid()}))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def close():
        stop.wait()
        server.shutdown()
        server.server_close()
        (project.state_dir / "endpoint.json").unlink(missing_ok=True)
    threading.Thread(target=close, daemon=True).start()
    return url
