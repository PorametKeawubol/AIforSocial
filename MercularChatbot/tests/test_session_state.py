"""Conversation-state expiry and context contracts."""

from types import SimpleNamespace

from session_state import RecentProducts, RecentQueries, RecentWebhookEvents


def test_recent_state_expires_and_failed_event_can_be_retried() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    events = RecentWebhookEvents(ttl_seconds=10, clock=clock)
    queries = RecentQueries(ttl_seconds=10, clock=clock)
    products = RecentProducts(ttl_seconds=10, clock=clock)

    assert events.claim("evt-1") is True
    assert events.claim("evt-1") is False
    events.release("evt-1")
    assert events.claim("evt-1") is True

    queries.remember("user-1", "query")
    products.remember_results(
        "user-1",
        [SimpleNamespace(id="1"), SimpleNamespace(id="1"), SimpleNamespace(id="2")],
    )
    products.focus("user-1", "2")
    assert queries.get("user-1") == "query"
    assert products.get("user-1") == (("1", "2"), "2")

    now[0] = 111.0
    assert events.claim("evt-1") is True
    assert queries.get("user-1") is None
    assert products.get("user-1") == ((), "")
