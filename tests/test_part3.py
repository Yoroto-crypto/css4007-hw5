import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")

from langchain_core.messages import AIMessage  # noqa: E402

from helpdesk.agent import text_of  # noqa: E402
from helpdesk.main import Runtime  # noqa: E402
from helpdesk.memory import UserMemory  # noqa: E402
from helpdesk.tools import Context, make_tools  # noqa: E402


class ScriptedModel:
    """Fake model that also records every message list it was given (self.seen)."""

    def __init__(self, script):
        self.script, self.calls, self.seen = script, 0, []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        text, calls = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return AIMessage(content=text, tool_calls=[
            {"name": n, "args": a, "id": f"call_{self.calls}_{i}"} for i, (n, a) in enumerate(calls)])


def all_text(messages):
    return " ".join(text_of(m.content) for m in messages)


# ---- the memory store alone ----

def test_memory_survives_reopen(tmp_path):
    m1 = UserMemory(tmp_path / "m.sqlite")
    m1.save("a", "language", "Kazakh")
    m1.close()
    assert UserMemory(tmp_path / "m.sqlite").load("a") == {"language": "Kazakh"}


def test_memory_is_isolated_between_users(tmp_path):
    m = UserMemory(tmp_path / "m.sqlite")
    m.save("a", "language", "Kazakh")
    assert m.load("b") == {}


def test_memory_overwrites_same_name(tmp_path):
    m = UserMemory(tmp_path / "m.sqlite")
    m.save("a", "language", "Kazakh")
    m.save("a", "language", "English")
    assert m.load("a") == {"language": "English"}


# ---- the remember_preference tool ----

def test_tool_saves_for_current_user_only(tmp_path):
    m = UserMemory(tmp_path / "m.sqlite")
    ctx = Context(user_id="a", tickets_dir=tmp_path)
    tool = {t.name: t for t in make_tools(ctx, m)}["remember_preference"]
    assert tool.invoke({"name": "language", "value": "Kazakh"})["status"] == "saved"
    assert m.load("a") == {"language": "Kazakh"} and m.load("b") == {}


def test_tool_refuses_secrets(tmp_path):
    m = UserMemory(tmp_path / "m.sqlite")
    tool = {t.name: t for t in make_tools(Context(user_id="a"), m)}["remember_preference"]
    assert tool.invoke({"name": "password", "value": "hunter2"})["status"] == "refused"
    assert m.load("a") == {}


def test_no_memory_tool_without_store():
    assert "remember_preference" not in {t.name for t in make_tools(Context())}


# ---- graph + SQLite, fake model ----

def test_conversation_survives_restart(tmp_path):
    model1 = ScriptedModel([("Noted.", [])])
    rt1 = Runtime("a", "t1", "normal", model1, storage_dir=tmp_path)
    rt1.ask("I have a Wi-Fi problem. I enabled it.")
    rt1.close()

    model2 = ScriptedModel([("You reported Wi-Fi.", [])])
    rt2 = Runtime("a", "t1", "normal", model2, storage_dir=tmp_path)  # a "new launch"
    result = rt2.ask("Which problem did we discuss?")
    rt2.close()
    assert "I enabled it" in all_text(model2.seen[-1])  # came from the saved thread, not from input
    assert "earlier messages" in result["sources"]["conversation_state"]


def test_preference_crosses_threads_but_not_users(tmp_path):
    save = ScriptedModel([("", [("remember_preference", {"name": "language", "value": "Kazakh"})]),
                          ("Done.", [])])
    rt = Runtime("a", "thread-1", "normal", save, storage_dir=tmp_path)
    rt.ask("Please remember that I prefer replies in Kazakh.")
    rt.close()

    same_user = ScriptedModel([("ok", [])])
    rt = Runtime("a", "thread-2", "normal", same_user, storage_dir=tmp_path)
    result = rt.ask("Which language did I ask for?")
    rt.close()
    assert "Kazakh" in text_of(same_user.seen[-1][0].content)  # system prompt carries user memory
    assert result["sources"]["user_memory_loaded"] == {"language": "Kazakh"}
    assert "earlier" not in result["sources"]["conversation_state"]  # new thread: no replay

    other_user = ScriptedModel([("ok", [])])
    rt = Runtime("b", "thread-3", "normal", other_user, storage_dir=tmp_path)
    result = rt.ask("Have I told you a preferred language?")
    rt.close()
    sent = all_text(other_user.seen[-1])
    assert "Kazakh" not in sent                 # the prompt itself must not hint at the scenario
    assert "nothing saved yet" in sent          # memory section is empty for this user
    assert result["sources"]["user_memory_loaded"] == {}
