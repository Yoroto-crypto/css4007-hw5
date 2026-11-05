"""Part 2 tools. Each tool is an ordinary Python function wrapped with @tool,
so it can be called directly (tool.invoke({...})) without any model or graph.

Tools are created by make_tools(ctx) so they know the selected status profile,
user id, thread id and where to save tickets.
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from helpdesk.classify import DATA_DIR, ROOT, load_helpdesk

VALID_STATUSES = {"operational", "outage", "unknown", "not_checked"}
SECRET_PATTERN = re.compile(
    r"(password|passcode|recovery code|api[ _-]?key)\s*(is|:|=)\s*\S+", re.IGNORECASE)


@dataclass
class Context:
    user_id: str = "demo-user-a"
    thread_id: str = "thread-1"
    status_profile: str = "normal"
    tickets_dir: Path = ROOT / "tickets"
    tickets_created: list = field(default_factory=list)  # filled by prepare_ticket
    memory_writes: list = field(default_factory=list)    # filled by remember_preference


def load_status_profile(name: str) -> dict:
    with open(DATA_DIR / "service_status.json", encoding="utf-8") as f:
        profiles = json.load(f)["profiles"]
    if name not in profiles:
        raise ValueError(f"unknown status profile: {name}. Available: {list(profiles)}")
    return profiles[name]


def make_tools(ctx: Context, memory=None):
    @tool
    def find_instructions(topic: str) -> dict:
        """Look up the official local troubleshooting guide for a topic.
        topic must be 'wifi' or 'password'. Returns the title, steps and instruction_id."""
        guides = load_helpdesk()["instructions"]
        key = topic.strip().lower()
        if key not in guides:
            return {"status": "not_found", "topic": topic, "available_topics": list(guides)}
        return {"status": "found", "topic": key, **guides[key]}

    @tool
    def check_service_status(service: str) -> dict:
        """Check the current status of a service in the demo status source.
        service must be 'wifi' or 'account_recovery' (password recovery).
        status is one of: operational, outage, unknown (status source unavailable)."""
        profile = load_status_profile(ctx.status_profile)
        key = service.strip().lower()
        if key not in profile:
            return {"status": "not_found", "service": service, "available_services": list(profile)}
        return {"service": key, "status_profile": ctx.status_profile, **profile[key]}

    @tool
    def prepare_ticket(category: str, summary: str, steps_tried: list[str],
                       service_status: str = "not_checked") -> dict:
        """Prepare a LOCAL ticket draft file. Nothing is sent anywhere.
        Only call this when the USER has stated both the problem (summary) and the steps
        already tried (steps_tried). Never invent either. category: 'wifi', 'password'
        or 'other'. service_status: 'operational', 'outage', 'unknown' or 'not_checked'
        (use the result of check_service_status, or 'not_checked' if it was not called)."""
        steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]
        missing = []
        if not summary or not summary.strip():
            missing.append("problem description")
        if not steps:
            missing.append("steps already tried")
        if missing:
            return {"status": "missing_information", "missing": missing,
                    "instruction": "Ask the user for the missing details. Do not invent them."}
        if service_status not in VALID_STATUSES:
            return {"status": "error", "message": f"service_status must be one of {sorted(VALID_STATUSES)}"}
        if SECRET_PATTERN.search(summary) or any(SECRET_PATTERN.search(s) for s in steps):
            return {"status": "refused",
                    "message": "The text seems to contain a password, key or recovery code. "
                               "Ask the user to remove it. Secrets must not be stored."}

        with open(DATA_DIR / "ticket_template.json", encoding="utf-8") as f:
            ticket = dict(json.load(f)["ticket"])
        guides = load_helpdesk()["instructions"]
        ticket_id = f"TKT-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        ticket.update({
            "ticket_id": ticket_id,
            "status": "draft",
            "user_id": ctx.user_id,
            "thread_id": ctx.thread_id,
            "category": category,
            "summary": summary.strip(),
            "steps_tried": steps,
            "service_status": service_status,
            "status_profile": ctx.status_profile,
            "reference_ids": [guides[category]["instruction_id"]] if category in guides else [],
        })
        ctx.tickets_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.tickets_dir / f"{ticket_id}.json"
        path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            shown = str(path.relative_to(ROOT))
        except ValueError:
            shown = str(path)
        ctx.tickets_created.append(shown)
        return {"status": "draft_saved", "ticket_file": shown, "ticket": ticket,
                "note": "Local draft only. Nothing was sent."}

    tools = [find_instructions, check_service_status, prepare_ticket]

    if memory is not None:
        @tool
        def remember_preference(name: str, value: str) -> dict:
            """Save a preference of the CURRENT user so it is remembered in future conversations,
            for example name='language', value='French'. Only call this when the user asks you to
            remember something. Never save passwords, API keys or recovery codes."""
            name, value = (name or "").strip().lower(), (value or "").strip()
            if not name or not value:
                return {"status": "error", "message": "name and value are both required"}
            if len(name) > 40 or len(value) > 200:
                return {"status": "error", "message": "name or value is too long"}
            secret_words = ("password", "passcode", "secret", "token", "key", "code")
            if any(w in name for w in secret_words) or SECRET_PATTERN.search(f"{name}: {value}"):
                return {"status": "refused",
                        "message": "Secrets such as passwords, keys or codes are never saved."}
            memory.save(ctx.user_id, name, value)  # always the current user: the model cannot choose
            ctx.memory_writes.append({"name": name, "value": value})
            return {"status": "saved", "name": name, "value": value}

        tools.append(remember_preference)

    return tools
