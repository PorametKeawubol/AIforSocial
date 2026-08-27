"""Thread-safe, bounded conversation state used by the LINE webhook."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class RecentWebhookEvents:
    """Protect the webhook from processing the same LINE delivery twice."""

    def __init__(
        self,
        ttl_seconds: float = 600,
        max_entries: int = 2_000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._events: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, event_id: str) -> bool:
        """Return false when an unexpired delivery ID was already claimed."""

        if not event_id:
            return True
        now = self._clock()
        with self._lock:
            self._prune(now)
            if event_id in self._events:
                return False
            while len(self._events) >= self.max_entries:
                self._events.pop(next(iter(self._events)))
            self._events[event_id] = now + self.ttl_seconds
            return True

    def release(self, event_id: str) -> None:
        """Allow LINE redelivery when no reply could be submitted."""

        if not event_id:
            return
        with self._lock:
            self._events.pop(event_id, None)

    def _prune(self, now: float) -> None:
        for key, expiry in list(self._events.items()):
            if expiry <= now:
                self._events.pop(key, None)


class RecentQueries:
    """Remember the last query so follow-up commands can reuse its filters."""

    def __init__(
        self,
        ttl_seconds: float = 1_800,
        max_entries: int = 2_000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._queries: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def remember(self, user_id: str, parsed_command: Any) -> None:
        if not user_id:
            return
        now = self._clock()
        with self._lock:
            self._prune(now)
            while len(self._queries) >= self.max_entries:
                self._queries.pop(next(iter(self._queries)))
            self._queries[user_id] = (now + self.ttl_seconds, parsed_command)

    def get(self, user_id: str) -> Any | None:
        if not user_id:
            return None
        now = self._clock()
        with self._lock:
            self._prune(now)
            value = self._queries.get(user_id)
            return value[1] if value else None

    def _prune(self, now: float) -> None:
        for key, (expiry, _value) in list(self._queries.items()):
            if expiry <= now:
                self._queries.pop(key, None)


class RecentProducts:
    """Remember shown products and the exact product opened by each user."""

    def __init__(
        self,
        ttl_seconds: float = 1_800,
        max_entries: int = 2_000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._values: dict[str, tuple[float, tuple[str, ...], str]] = {}
        self._lock = threading.Lock()

    def remember_results(self, user_id: str, products: list[object]) -> None:
        if not user_id:
            return
        identifiers = tuple(
            dict.fromkeys(
                str(identifier)
                for product in products
                if (identifier := getattr(product, "id", ""))
            )
        )
        now = self._clock()
        with self._lock:
            self._prune(now)
            previous = self._values.get(user_id)
            focus = previous[2] if previous else ""
            self._make_room(user_id)
            self._values[user_id] = (now + self.ttl_seconds, identifiers, focus)

    def focus(self, user_id: str, product_id: str) -> None:
        if not user_id or not product_id:
            return
        now = self._clock()
        with self._lock:
            self._prune(now)
            previous = self._values.get(user_id)
            identifiers = previous[1] if previous else ()
            self._make_room(user_id)
            self._values[user_id] = (
                now + self.ttl_seconds,
                identifiers,
                str(product_id),
            )

    def get(self, user_id: str) -> tuple[tuple[str, ...], str]:
        if not user_id:
            return (), ""
        now = self._clock()
        with self._lock:
            self._prune(now)
            value = self._values.get(user_id)
            return (value[1], value[2]) if value else ((), "")

    def _make_room(self, user_id: str) -> None:
        if user_id not in self._values:
            while len(self._values) >= self.max_entries:
                self._values.pop(next(iter(self._values)))

    def _prune(self, now: float) -> None:
        for key, (expiry, _results, _focus) in list(self._values.items()):
            if expiry <= now:
                self._values.pop(key, None)


__all__ = ["RecentProducts", "RecentQueries", "RecentWebhookEvents"]
