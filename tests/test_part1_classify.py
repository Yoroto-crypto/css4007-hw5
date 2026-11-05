import pytest

from helpdesk.classify import classify


@pytest.mark.parametrize("message,expected", [
    ("I cannot connect to Wi-Fi.", "wifi"),
    ("I cannot connect to WIFI", "wifi"),
    ("I forgot my PASSWORD.", "password"),
    ("My laptop screen is broken.", "other"),
    ("My Wi-Fi password does not work.", "wifi"),   # Wi-Fi rule wins in the original graph
])
def test_original_rules(message, expected):
    assert classify(message) == expected


def test_both_topics_detected_in_changed_graph():
    assert classify("My Wi-Fi password does not work.", detect_both=True) == "both"
    assert classify("I forgot my password", detect_both=True) == "password"
