"""The pages' login: the project's web token, ``<state>/web.token`` (owner-only, made the first
time). Adapted from the old Orchestra's ``web/auth.py`` without Starlette.

- **Browsers** log in once at ``/login`` with the token (``orchestra run --web`` prints a link
  carrying it after ``#``, which the page posts and takes out of the address bar) and keep a
  session cookie, signed with a key derived from the token (HMAC-SHA256), HttpOnly,
  SameSite=Strict, valid for ``SESSION_S``. **Programs** send ``Authorization: Bearer <token>``.
- **Guessing** is limited per peer address: ``FAIL_MAX`` failures in ``FAIL_WINDOW_S``, then 429.
- A POST must come from the page's own origin (``Origin`` or ``Referer`` naming this host)."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import deque
from urllib.parse import urlsplit

SESSION_S = 14 * 24 * 3600
FAIL_MAX = 5
FAIL_WINDOW_S = 60.0


def token(project) -> str:
    path = project.path("web.token")
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    return path.read_text().strip()


def cookie_name(project) -> str:
    return "orchestra_" + "".join(c if c.isalnum() else "_" for c in project.name)


class Secret:
    def __init__(self, project):
        self.value = token(project)
        self.key = hmac.new(self.value.encode(), b"orchestra web session v1", hashlib.sha256).digest()

    def matches(self, given: str) -> bool:
        return hmac.compare_digest(given.strip().encode(), self.value.encode())

    def _sign(self, body: str) -> str:
        return hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()

    def session(self, now: float | None = None) -> str:
        body = f"v1.{int((now or time.time()) + SESSION_S)}"
        return f"{body}.{self._sign(body)}"

    def valid_session(self, value: str | None, now: float | None = None) -> bool:
        parts = (value or "").split(".")
        if len(parts) != 3 or parts[0] != "v1" or not parts[1].isdigit():
            return False
        return hmac.compare_digest(parts[2], self._sign(f"v1.{parts[1]}")) and int(parts[1]) > (now or time.time())


class Failures:
    """Failed token checks per peer address, in a sliding window."""

    def __init__(self, most: int = FAIL_MAX, window_s: float = FAIL_WINDOW_S):
        self.most, self.window_s = most, window_s
        self._by: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def wait(self, peer: str, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._by.get(peer)
            while q and q[0] <= now - self.window_s:
                q.popleft()
            return max(0.0, q[0] + self.window_s - now) if q and len(q) >= self.most else 0.0

    def fail(self, peer: str, now: float | None = None) -> None:
        with self._lock:
            if len(self._by) > 10_000:
                self._by.pop(next(iter(self._by)))
            self._by.setdefault(peer, deque(maxlen=self.most)).append(time.monotonic() if now is None else now)


def same_origin(headers) -> bool:
    if headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
        return False
    origin = headers.get("Origin") or headers.get("Referer")
    if not origin:
        return True
    return origin != "null" and urlsplit(origin).netloc.lower() == (headers.get("Host") or "").lower()


def safe_next(value: str | None) -> str:
    v = value or "/"
    return v if v.startswith("/") and not v.startswith(("//", "/\\", "/login", "/logout")) else "/"
