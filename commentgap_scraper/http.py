from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable


class HttpFailure(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class RateLimiter:
    """Thread-safe limiter measured between request start times."""

    def __init__(self, interval_seconds: float = 1.0, clock: Callable[[], float] = time.monotonic):
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._last_started: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_started is not None:
                delay = self.interval_seconds - (now - self._last_started)
                if delay > 0:
                    time.sleep(delay)
            self._last_started = self._clock()


@dataclass(slots=True)
class HttpClient:
    user_agent: str
    interval_seconds: float = 1.0
    timeout: float = 30.0
    max_retries: int = 5
    _limiter: RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.interval_seconds)

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
    ) -> bytes:
        body = None if json_body is None else json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json,text/html,application/xml,text/xml;q=0.9,*/*;q=0.8",
            "User-Agent": self.user_agent,
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable:
                    raise HttpFailure(
                        f"HTTP {exc.code} for {url}", status=exc.code, retryable=False
                    ) from exc
                last_error = exc
                retry_after = self._retry_after_seconds(exc.headers.get("Retry-After"))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                retry_after = None

            if attempt == self.max_retries:
                break
            backoff = min(60.0, 2.0**attempt) + random.uniform(0.0, 0.5)
            time.sleep(max(backoff, retry_after or 0.0))

        raise HttpFailure(
            f"request failed after {self.max_retries + 1} attempts for {url}: {last_error}",
            status=getattr(last_error, "code", None),
            retryable=True,
        ) from last_error

    def get_text(self, url: str) -> str:
        return self.request(url).decode("utf-8", errors="replace")

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = self.request(url, method="POST", json_body=payload)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HttpFailure(f"invalid JSON returned by {url}") from exc
        if result.get("errors"):
            messages = "; ".join(str(error.get("message", error)) for error in result["errors"])
            raise HttpFailure(f"GraphQL error from {url}: {messages}")
        return result
