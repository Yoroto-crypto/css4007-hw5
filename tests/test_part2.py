import json

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

from helpdesk.agent import build_agent_graph, run_turn  # noqa: E402
from helpdesk.tools import Context, make_tools  # noqa: E402


class ScriptedModel:
    """Fake model: replays a script of (text, [tool calls]); the last item repeats."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        text, calls = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        tool_calls = [{"name": n, "args": a, "id": f"call_{self.calls}_{i}"}
                      for i, (n, a) in enumerate(calls)]
        return AIMessage(content=text, tool_calls=tool_calls)


def make_ctx(tmp_path, profile="normal"):
    return Context(user_id="u", thread_id="t", status_profile=profile, tickets_dir=tmp_path)


def tools_of(ctx):
    return {t.name: t for t in make_tools(ctx)}


# ---- tools alone (no model, no graph) ----

@pytest.mark.parametrize("profile,expected", [
    ("normal", "operational"), ("wifi_outage", "outage"), ("status_unavailable", "unknown")])
def test_status_follows_selected_profile(tmp_path, profile, expected):
    t = tools_of(make_ctx(tmp_path, profile))
    assert t["check_service_status"].invoke({"service": "wifi"})["status"] == expected


def test_recovery_outage_only_affects_account_recovery(tmp_path):
    t = tools_of(make_ctx(tmp_path, "recovery_outage"))
    assert t["check_service_status"].invoke({"service": "account_recovery"})["status"] == "outage"
    assert t["check_service_status"].invoke({"service": "wifi"})["status"] == "operational"


def test_unknown_topic_is_not_invented(tmp_path):
    t = tools_of(make_ctx(tmp_path))
    assert t["find_instructions"].invoke({"topic": "printer"})["status"] == "not_found"


def test_ticket_refuses_missing_details_and_writes_nothing(tmp_path):
    t = tools_of(make_ctx(tmp_path))
    r = t["prepare_ticket"].invoke({"category": "wifi", "summary": "", "steps_tried": []})
    assert r["status"] == "missing_information"
    assert list(tmp_path.iterdir()) == []


def test_ticket_uses_template_and_user_steps(tmp_path):
    ctx = make_ctx(tmp_path)
    r = tools_of(ctx)["prepare_ticket"].invoke({
        "category": "wifi", "summary": "Cannot connect", "steps_tried": ["enabled Wi-Fi"],
        "service_status": "operational"})
    saved = json.loads(next(tmp_path.iterdir()).read_text(encoding="utf-8"))
    assert saved["status"] == "draft" and saved["steps_tried"] == ["enabled Wi-Fi"]
    assert saved["reference_ids"] == ["GUIDE-WIFI"] and r["status"] == "draft_saved"


def test_ticket_refuses_secrets(tmp_path):
    r = tools_of(make_ctx(tmp_path))["prepare_ticket"].invoke({
        "category": "password", "summary": "my password: hunter2", "steps_tried": ["x"]})
    assert r["status"] == "refused" and list(tmp_path.iterdir()) == []


# ---- loop plumbing with a FAKE model (checks the graph, not the real model's judgement) ----

def test_loop_passes_tool_result_back_to_model(tmp_path):
    ctx = make_ctx(tmp_path, "wifi_outage")
    model = ScriptedModel([("", [("check_service_status", {"service": "wifi"})]),
                           ("Wi-Fi is down; no ticket prepared.", [])])
    result = run_turn(build_agent_graph(ctx, llm=model, checkpointer=None), ctx, "check wifi")
    assert result["reply"] == "Wi-Fi is down; no ticket prepared."
    assert "outage" in result["steps"][1]["tool_results"][0]["result"]
    assert result["tickets"] == [] and not result["loop_limit_reached"]


def test_loop_limit_stops_endless_tool_calls(tmp_path):
    ctx = make_ctx(tmp_path)
    model = ScriptedModel([("", [("check_service_status", {"service": "wifi"})])])  # forever
    result = run_turn(build_agent_graph(ctx, llm=model, max_model_calls=3), ctx, "loop")
    assert result["loop_limit_reached"] and model.calls == 3
