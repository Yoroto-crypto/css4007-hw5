import pytest

pytest.importorskip("langgraph")

from helpdesk.part1_graph import build_graph, run_with_trace  # noqa: E402


def nodes(variant, message):
    _, steps = run_with_trace(build_graph(variant), message)
    return [s["node"] for s in steps]


def test_wifi_route():
    assert nodes("original", "I cannot connect to Wi-Fi.") == ["read_request", "wifi_help"]


def test_password_route_case_insensitive():
    assert nodes("original", "I forgot my PASSWORD.") == ["read_request", "password_help"]


def test_other_route():
    assert nodes("original", "My laptop screen is broken.") == ["read_request", "other_help"]


def test_two_topics_original_goes_to_wifi():
    assert nodes("original", "My Wi-Fi password does not work.") == ["read_request", "wifi_help"]


def test_two_topics_changed_graph_clarifies():
    assert nodes("with_clarification", "My Wi-Fi password does not work.") == [
        "read_request", "clarify_problem"]


def test_message_is_kept_and_reply_is_set():
    state, _ = run_with_trace(build_graph("original"), "I cannot connect to Wi-Fi.")
    assert state["message"] == "I cannot connect to Wi-Fi."
    assert "GUIDE-WIFI" in state["reply"]
