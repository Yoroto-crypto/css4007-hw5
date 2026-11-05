"""Plain-Python helpers (no LangGraph): data loading and keyword classification.

Keeping this separate means the routing rules can be unit-tested on their own.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_helpdesk() -> dict:
    with open(DATA_DIR / "helpdesk.json", encoding="utf-8") as f:
        return json.load(f)


def mentions(message: str, topic: str, helpdesk: dict | None = None) -> bool:
    """True if the message contains any alias of the topic (case-insensitive)."""
    helpdesk = helpdesk or load_helpdesk()
    text = message.lower()
    return any(alias in text for alias in helpdesk["topic_aliases"][topic])


def classify(message: str, detect_both: bool = False) -> str:
    """Return 'wifi', 'password', 'other' (or 'both' if detect_both=True).

    Rules from the brief: wifi/wi-fi first, otherwise password, otherwise other.
    With detect_both=True (the changed graph) a message with both topics -> 'both'.
    """
    helpdesk = load_helpdesk()
    has_wifi = mentions(message, "wifi", helpdesk)
    has_password = mentions(message, "password", helpdesk)
    if detect_both and has_wifi and has_password:
        return "both"
    if has_wifi:
        return "wifi"
    if has_password:
        return "password"
    return "other"


def format_instructions(topic: str) -> str:
    """Build a readable reply from helpdesk.json for 'wifi' or 'password'."""
    guide = load_helpdesk()["instructions"][topic]
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(guide["steps"], start=1))
    return f"{guide['title']} [{guide['instruction_id']}]\n{steps}"
