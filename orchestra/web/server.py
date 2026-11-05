"""The pages' HTTP server (stdlib): every route but ``/health``, ``/login`` and ``/static/`` needs the
login (``auth``); a browser without it is sent to ``/login``, a program gets 401. ``/api/<page>``
returns a page's data as JSON. Read only: the only POSTs are ``/login`` and ``/logout``."""

from __future__ import annotations

import json
import mimetypes
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from . import data, pages
from .auth import SESSION_S, Failures, Secret, cookie_name, safe_next, same_origin

STATIC = Path(__file__).parent / "static"
LOGIN_BODY_MAX = 8192


def _int(q: dict, k: str, default: int = 0) -> int:
    try:
        return max(0, int((q.get(k) or [default])[0]))
    except ValueError:
        return default


def views(project):
    """path -> (title, data(query), html(data))."""
    return {
        "/": ("Dashboard", lambda q: data.dashboard(project), pages.dashboard),
        "/items": ("Items", lambda q: data.items(project, _int(q, "page"), 25, (q.get("closed") or ["0"])[0] == "1", (q.get("kind") or [""])[0] or None), pages.items),
        "/machines": ("Machines", lambda q: data.machines(project), pages.machines),
        "/dynamics": ("Dynamics", lambda q: data.dynamics(project), pages.dynamics),
        "/core": ("Core", lambda q: data.core(project), pages.core),
    }


def handler(project):
    secret, failures, cookie = Secret(project), Failures(), cookie_name(project)
    routes = views(project)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes | str, ctype: str = "text/html; charset=utf-8", headers: dict | None = None):
            raw = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(raw)

        def _json(self, code: int, body) -> None:
            self._send(code, json.dumps(body, default=str), "application/json")

        def _peer(self) -> str:
            return self.client_address[0]

        def _authorized(self) -> bool | None:
            """True, False, or None when the peer is refused for guessing."""
            scheme, _, given = (self.headers.get("Authorization") or "").partition(" ")
            if scheme.lower() == "bearer":
                if failures.wait(self._peer()):
                    return None
                if secret.matches(given):
                    return True
                failures.fail(self._peer())
                return False
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get("Cookie") or "")
            except Exception:
                return False
            return secret.valid_session(jar[cookie].value if cookie in jar else None)

        def _cookie(self, value: str, max_age: int) -> str:
            return f"{cookie}={value}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Strict"

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            if u.path == "/health":
                return self._json(200, {"ok": True})
            if u.path == "/login":
                return self._send(200, pages.login(safe_next((q.get("next") or ["/"])[0]), ""))
            if u.path.startswith("/static/"):
                f = (STATIC / u.path.removeprefix("/static/")).resolve()
                if STATIC.resolve() not in f.parents or not f.is_file():
                    return self._send(404, "not found", "text/plain")
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            ok = self._authorized()
            if ok is None:
                return self._json(429, {"error": "too many wrong tokens; try again later"})
            api = u.path.startswith("/api/")
            if not ok:
                if api or "text/html" not in (self.headers.get("Accept") or ""):
                    return self._json(401, {"error": "log in at /login, or send Authorization: Bearer <token>"})
                return self._send(303, "", headers={"Location": f"/login?next={quote(self.path, safe='')}"})
            path = "/" if u.path in ("/api/dashboard", "/api/") else u.path.removeprefix("/api")
            if path == "/items" and q.get("show"):
                found = data.item(project, q["show"][0])
                if found is None:
                    return self._send(404, pages.layout(project, "/items", "Items", "<p>no such item</p>"))
                if api:
                    return self._json(200, found)
                return self._send(200, pages.layout(project, "/items", found["subject"], pages.item(found)))
            if path not in routes:
                return self._send(404, pages.layout(project, "", "Not found", "<p>no such page</p>"))
            title, get, html = routes[path]
            d = get(q)
            if api:
                return self._json(200, d)
            embed = d if path in ("/machines", "/dynamics") else None
            self._send(200, pages.layout(project, path, title, html(d), embed))

        def do_POST(self):
            u = urlsplit(self.path)
            if not same_origin(self.headers):
                return self._json(403, {"error": "refused: not sent from this site's own pages"})
            if u.path == "/logout":
                return self._send(303, "", headers={"Location": "/login", "Set-Cookie": self._cookie("", 0)})
            if u.path != "/login":
                return self._json(405, {"error": "the pages are read only"})
            n = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(min(n, LOGIN_BODY_MAX)).decode("utf-8", "replace"))
            nxt = safe_next((form.get("next") or ["/"])[0])
            if wait := failures.wait(self._peer()):
                return self._send(429, pages.login(nxt, f"Too many wrong tokens. Try again in {int(wait) + 1} s."))
            if not secret.matches((form.get("token") or [""])[0]):
                failures.fail(self._peer())
                return self._send(401, pages.login(nxt, "That is not this project's token."))
            self._send(303, "", headers={"Location": nxt, "Set-Cookie": self._cookie(secret.session(), SESSION_S)})

    return Handler
