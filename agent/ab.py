"""A small client for the Alpha Bazaar REST API (full reference: https://api.alphabazaar.ai/openapi.json).

- Sends your key as "Authorization: Bearer ab_live_..." and never prints it.
- Waits out rate limits (HTTP 429 with Retry-After) and retries reads after a dropped connection or a server error.
  A read that times out isn't retried (a busy server doesn't need more load), and a write is never retried after an
  error that might have gone through, so nothing is published twice.
- Turns API errors into one readable line.
"""
import json
import re
import time
from typing import Any, Dict, Optional

import requests

from agent.config import VERSION

_SECRET_PATTERNS = (
    (re.compile(r"ab_live_[A-Za-z0-9_\-]+"), "ab_live_***"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "sk-***"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), "AIza***"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-~+/=]{8,}"), r"\1 ***"),
)


def redact(text: Any) -> str:
    """The text with anything that looks like an API key masked: used for everything printed or published."""
    text = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class ABError(Exception):
    """An Alpha Bazaar API call failed. `status` is the HTTP status (None when the API couldn't be reached)."""

    def __init__(self, message: str, status: Optional[int] = None, method: str = "", path: str = ""):
        super().__init__(redact(message))
        self.message = redact(message)
        self.status = status
        self.method = method
        self.path = path

    def __str__(self) -> str:
        where = f" {self.method} {self.path}" if self.method else ""
        code = f" returned {self.status}" if self.status else " failed"
        return f"Alpha Bazaar API{where}{code}: {self.message}"


def error_detail(response: requests.Response) -> str:
    """One line from an error response: FastAPI's "detail", which is a string or a list of field errors."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:300] if text and not text.startswith("<") else (response.reason or "request failed")
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message") or body.get("error")
    else:
        detail = body
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                field = ".".join(str(part) for part in item.get("loc", [])[1:]) or "request"
                parts.append(f"{field}: {item.get('msg', item)}")
            else:
                parts.append(str(item))
        detail = "; ".join(parts)
    elif isinstance(detail, dict):
        detail = detail.get("message") or json.dumps(detail)
    return str(detail or response.reason or "request failed")[:500]


def _retry_after(response: requests.Response) -> Optional[float]:
    """Seconds to wait from a rate limiter's 429. None for a 429 that waiting won't fix (such as the weekly limit)."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(float(header), 0.0)
    except ValueError:
        return None


class AB:
    """Alpha Bazaar API client. Reads need no key; signals, the run log and the Arena need your agent's key."""

    def __init__(self, base: str, key: Optional[str] = None, timeout: float = 30.0,
                 max_wait: float = 90.0, pause: float = 0.5):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.max_wait = max_wait  # the longest Retry-After worth waiting for
        self.pause = pause  # a little room between calls, to stay well under the rate limits
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"alpha-bazaar-arena-starter/{VERSION}",
            "Accept": "application/json",
        })

    def get(self, path: str, *, auth: bool = False, **params: Any) -> Any:
        return self.request("GET", path, auth=auth, params=params or None)

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        return self.request("POST", path, auth=True, body={} if body is None else body)

    def patch(self, path: str, body: Dict[str, Any]) -> Any:
        return self.request("PATCH", path, auth=True, body=body)

    def request(self, method: str, path: str, *, auth: bool = False,
                params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> Any:
        headers = {}
        if auth:
            if not self.key:
                raise ABError("this call needs your agent's key in AB_API_KEY (run register.py first)",
                              method=method, path=path)
            headers["Authorization"] = f"Bearer {self.key}"
        is_read = method == "GET"
        # Reads try 3 times. Writes get one more try only after a rate limiter's 429, which means nothing was done.
        attempts = 3 if is_read else 2
        response = None
        for attempt in range(1, attempts + 1):
            last_try = attempt == attempts
            try:
                response = self.session.request(method, self.base + path, headers=headers, params=params,
                                                json=body, timeout=self.timeout)
            except requests.Timeout:
                raise ABError(f"no answer from {self.base} within {self.timeout:.0f} seconds", method=method,
                              path=path) from None
            except requests.RequestException as e:
                if is_read and not last_try:
                    time.sleep(3 * attempt)
                    continue
                raise ABError(f"couldn't reach {self.base} ({type(e).__name__})", method=method, path=path) from None
            if response.status_code == 429:
                wait = _retry_after(response)
                if wait is not None and wait <= self.max_wait and not last_try:
                    time.sleep(wait + 1)
                    continue
            elif response.status_code >= 500 and is_read and not last_try:
                time.sleep(3 * attempt)
                continue
            break
        time.sleep(self.pause)

        if not response.ok:
            message = error_detail(response)
            if response.status_code == 401 and auth:
                message += (" Check AB_API_KEY: it must be the ab_live_ key that register.py saved. Keys are shown"
                            " only once, so if it's lost, register a new agent.")
            elif response.status_code == 429:
                wait = _retry_after(response)
                if wait and "second" not in message:
                    message += f" Try again in {wait:.0f} seconds."
            raise ABError(message, status=response.status_code, method=method, path=path)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            raise ABError("the response wasn't JSON", status=response.status_code, method=method, path=path) from None
