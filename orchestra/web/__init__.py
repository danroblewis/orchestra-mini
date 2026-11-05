"""The pages (SPEC §7), read only: Dashboard, Items, Machines, Dynamics and Core, behind the
project's token (``auth``). ``serve(project, stop, listen)`` is the seam ``orchestra run --web``
uses (``orchestra.hooks``): it prints a login link and serves until ``stop``."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

from .auth import token
from .server import handler


def make_server(project, listen: str = "127.0.0.1:8765") -> ThreadingHTTPServer:
    host, _, port = listen.rpartition(":")
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port or 0)), handler(project))
    server.daemon_threads = True
    return server


def serve(project, stop: threading.Event, listen: str = "127.0.0.1:8765") -> None:
    server = make_server(project, listen)
    host, port = server.server_address[:2]
    print(f"pages: http://{host}:{port}/login#token={token(project)}", flush=True)
    project.path("web.json").write_text(f'{{"url": "http://{host}:{port}"}}')
    threading.Thread(target=server.serve_forever, daemon=True).start()
    stop.wait()
    server.shutdown()
    server.server_close()
    (project.state_dir / "web.json").unlink(missing_ok=True)
