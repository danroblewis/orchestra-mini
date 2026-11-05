"""One request to a Prometheus-API store, as a busy managed store needs it asked. Copied from the
old Orchestra's ``sources/transport.py``, without budgets, deadlines and mTLS.

- Every request carries the store's auth (``headers``: ``token_env`` as ``Authorization: Bearer``,
  or ``auth: {header, token_env}`` as that header), read from the environment when it is sent, so a
  rotated token is picked up; takes a token from the store's rate limit (``limits``); and is logged
  (``reqlog``).
- **429, 502, 503, 504** are waited out and asked again, at most ``RETRIES`` times: after the
  server's ``Retry-After`` (believed up to ``MAX_WAIT``; a 429's holds every reader of the store),
  else an exponential pause with jitter. A dropped connection is asked again the same way.
- **422, or a 4xx that says "too many"**, is the store's size limit: ``TooBig``, never retried.
- Other 4xx and 5xx raise ``StoreError`` at once, named by the server and what it said, never a
  credential. Redirects are followed only to the same scheme, host and port."""

from __future__ import annotations

import email.utils
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import reqlog
from .guard import StoreError
from .limits import DailyCap, RateLimit

RETRY = {429, 502, 503, 504}
RETRIES = 4
BACKOFF = 1.0
MAX_WAIT = 30.0
TIMEOUT = 30.0
TOO_BIG = re.compile(r"(?i)too many|exceed|limit of|maximum number|max(imum)?[ _]?series|unique (time)?series")


class TooBig(StoreError):
    """The store refused a read for its size: ask for less, do not retry it."""


def headers(store) -> dict[str, str]:
    """The auth header the store's config names, from the environment; {} without one. Raises
    ``StoreError`` naming the variable (never its value) when it is not set."""
    header, env = "Authorization", store.token_env
    if store.auth is not None and store.auth.token_env:
        header, env = store.auth.header or "Authorization", store.auth.token_env
    if not env:
        return {}
    token = os.environ.get(env, "").strip()
    if not token:
        raise StoreError(f"store {store.name}: the environment variable {env} is not set")
    return {header: f"Bearer {token}" if header.lower() == "authorization" else token}


def retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header: a number of seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value.strip()).timestamp()
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, when - (time.time() if now is None else now))


def _origin(url: str) -> tuple:
    u = urllib.parse.urlsplit(url)
    return u.scheme.lower(), (u.hostname or "").lower(), u.port or {"http": 80, "https": 443}.get(u.scheme.lower())


class SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to the same scheme, host and port, so credentials never leave."""

    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        if _origin(target) != _origin(req.full_url):
            raise urllib.error.HTTPError(req.full_url, code, "redirected to another server; not followed", hdrs, fp)
        return super().redirect_request(req, fp, code, msg, hdrs, newurl)


class Transport:
    def __init__(self, project, store, limiter: RateLimit):
        self.project, self.store, self.limiter = project, store, limiter
        self.url = store.url.rstrip("/")
        self.requests = 0
        self._sleep = time.sleep
        self._opener = urllib.request.build_opener(SameOriginRedirects())

    def where(self) -> str:
        u = urllib.parse.urlsplit(self.url)
        return f"{u.scheme}://{u.hostname}{f':{u.port}' if u.port else ''}"

    def log(self, path: str, form: dict | None, began: float, *, data=None, error: str | None = None, sent: bool = True) -> None:
        q, win, step = reqlog.asked_about(form)
        series, points = reqlog.answer_size(data) if error is None else (None, None)
        reqlog.append(self.project, self.store.name, {
            "endpoint": path, "query": q, "window": win, **({"step": step} if step is not None else {}),
            "series": series, "points": points, "ms": round(1000 * (time.monotonic() - began), 1),
            **({"sent": False} if not sent else {}), **({"error": error} if error else {})})

    def call(self, path: str, form: dict | None = None, *, post: bool = True):
        began = time.monotonic()
        try:
            data = self._ask(path, form, post=post)
        except StoreError as e:
            self.log(path, form, began, error=str(e), sent=not isinstance(e, DailyCap))
            raise
        self.log(path, form, began, data=data)
        return data

    def _ask(self, path: str, form: dict | None, *, post: bool):
        url, body = self.url + path, None
        if form is not None and post:
            body = urllib.parse.urlencode(form, doseq=True).encode()
        elif form is not None:
            url += "?" + urllib.parse.urlencode(form, doseq=True)
        where = f"{self.where()}{path}"
        for attempt in range(RETRIES + 1):
            self.limiter.take(where)
            hdrs = {"Accept": "application/json", **headers(self.store)}
            if body is not None:
                hdrs["Content-Type"] = "application/x-www-form-urlencoded"
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if body is not None else "GET")
            self.requests += 1
            try:
                with self._opener.open(req, timeout=TIMEOUT) as r:
                    raw = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code in RETRY and attempt < RETRIES:
                    self._wait(e, attempt)
                    continue
                said = self._said(e)
                if e.code == 422 or (400 <= e.code < 500 and e.code not in (401, 403, 404, 429) and TOO_BIG.search(said)):
                    raise TooBig(f"{where}: {said}") from None
                raise StoreError(f"{where}: {said}") from None
            except (TimeoutError, ConnectionResetError, http.client.RemoteDisconnected, http.client.IncompleteRead) as e:
                if attempt < RETRIES:
                    self._pause(attempt)
                    continue
                raise StoreError(f"{where}: no answer ({type(e).__name__})") from None
            except OSError as e:
                raise StoreError(f"{where}: {getattr(e, 'reason', None) or e}") from None
            except ValueError:
                raise StoreError(f"{where}: invalid request header (a credential is malformed)") from None
        try:
            doc = json.loads(raw)
        except ValueError as e:
            raise StoreError(f"{where}: the answer is not JSON ({e})") from None
        if not isinstance(doc, dict) or doc.get("status") != "success":
            raise StoreError(f"{where}: {doc.get('error') if isinstance(doc, dict) else 'not a Prometheus answer'}")
        return doc.get("data")

    def _pause(self, attempt: int) -> None:
        full = min(MAX_WAIT, BACKOFF * 2 ** attempt)
        self._sleep(full / 2 + random.uniform(0, full / 2))

    def _wait(self, e: urllib.error.HTTPError, attempt: int) -> None:
        told = retry_after(e.headers.get("Retry-After") if e.headers else None)
        try:
            e.read()
        except (OSError, AttributeError):
            pass
        if told is None:
            return self._pause(attempt)
        wait = min(MAX_WAIT, told)
        if e.code == 429:
            self.limiter.hold(wait)
        self._sleep(wait + random.uniform(0, min(1.0, 0.2 * wait)))

    @staticmethod
    def _said(e: urllib.error.HTTPError) -> str:
        if e.code in (401, 403):
            return f"HTTP {e.code}: the server refused the credentials (check the token)"
        try:
            doc = json.loads(e.read())
            err = doc.get("error") if isinstance(doc, dict) else None
        except (ValueError, AttributeError, OSError):
            err = None
        return f"HTTP {e.code}" + (f": {err}" if err else "")
