"""Parts 2 and 3: model + tools loop in LangGraph, optionally with per-user memory.

Part 2 (no memory):  START -> agent <-> tools -> END
Part 3 (memory):     START -> load_user_memory -> agent <-> tools -> END
                     (agent --limit hit--> limit -> END)

Conversation state = LangGraph checkpointer, keyed by thread_id.
User memory        = UserMemory (SQLite), keyed by user_id.
"""
import json
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from helpdesk.classify import ROOT, load_helpdesk
from helpdesk.tools import Context, make_tools

MAX_MODEL_CALLS = 6  # loop limit: model calls per user message


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model_calls: int
    limit_reached: bool
    user_memory: dict  # loaded from the user-memory store at the start of every turn


BASE_PROMPT = """You are a university IT helpdesk assistant for a PRACTICE exercise. All data is fictional.
Your scope: {scope}. Politely say so if a request is outside this scope.

Rules:
1. Use tools. Never invent instructions, service status or ticket details.
2. To show troubleshooting steps call find_instructions (topic 'wifi' or 'password').
3. If the user asks whether a service is down, or makes a ticket conditional on service status, call
   check_service_status first ('wifi' for Wi-Fi, 'account_recovery' for password recovery).
4. If the status is 'outage' and the user wanted a ticket only if the service is operating, do NOT call
   prepare_ticket. Explain the outage. Do not invent a repair time.
5. If the status is 'unknown', say you cannot confirm the service is operating and ask whether the user
   wants a draft ticket explaining that the status could not be checked. Do not create it until they agree.
6. prepare_ticket needs the problem and the steps already tried, both stated by the user. If either is
   missing, ask the user. Do not call the tool and do not guess.
7. steps_tried must contain only steps the user said they did. Do not add steps.
8. Tickets are local drafts only; nothing is sent. Tell the user the ticket file path.
9. Never ask for or store passwords, API keys or recovery codes.
Keep replies short."""

MEMORY_RULES = """
10. If the user asks you to remember a preference (for example the reply language), call
    remember_preference (for example name='language', value='French'). Do not save anything else.
11. If a saved language preference exists, write your replies in that language.
12. Questions about what the user said earlier in THIS conversation: answer from the conversation
    history. Questions about preferences from past conversations: answer ONLY from the saved memory
    below. If nothing relevant is in the history or the saved memory, say that nothing has been
    recorded for this user. Never guess.

Saved memory for this user (from earlier conversations; it belongs to this user only):
{memory}"""


def build_system_prompt(user_memory: dict | None = None, memory_enabled: bool = False) -> str:
    prompt = BASE_PROMPT.format(scope="; ".join(load_helpdesk()["scope"]))
    if memory_enabled:
        shown = json.dumps(user_memory, ensure_ascii=False) if user_memory else "nothing saved yet"
        prompt += MEMORY_RULES.format(memory=shown)
    return prompt


class ModelUnavailable(Exception):
    """The provider is overloaded, rate-limited or has no quota for this model."""


def invoke_with_retry(model, messages, attempts: int = 6, first_delay: float = 3.0,
                      max_delay: float = 30.0):
    """Call the model; wait and retry on 503/429. Raise ModelUnavailable when it keeps failing."""
    for attempt in range(1, attempts + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - anything not temporary is re-raised below
            text = str(exc)
            # A zero or per-day quota will not recover in seconds: do not wait.
            if "limit: 0" in text or "PerDay" in text:
                raise ModelUnavailable("no free quota for this model on your project, or the "
                                       "daily quota is used up (see https://ai.dev/rate-limit)") from exc
            temporary = any(c in text for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not temporary:
                raise
            if attempt == attempts:
                raise ModelUnavailable(f"still unavailable after {attempts} attempts") from exc
            delay = min(first_delay * 2 ** (attempt - 1), max_delay)
            print(f"  [model busy, retry {attempt}/{attempts - 1} in {delay:.0f}s]")
            time.sleep(delay)


class FallbackModel:
    """Tries the primary model (with retries), then each fallback model in order."""

    def __init__(self, models, names):
        self.models, self.names = models, names

    def bind_tools(self, tools):
        return FallbackModel([m.bind_tools(tools) for m in self.models], self.names)

    def invoke(self, messages):
        for i, (name, model) in enumerate(zip(self.names, self.models)):
            try:
                return invoke_with_retry(model, messages)
            except ModelUnavailable as exc:
                if i == len(self.models) - 1:
                    raise
                print(f"  [model {name} unavailable: {exc}; switching to {self.names[i + 1]}]")


def get_model():
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv(ROOT / ".env")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            f"No API key found. Create the file {ROOT / '.env'} with a line "
            "GOOGLE_API_KEY=<your key from Google AI Studio> (no quotes), save it and rerun.")
    primary = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fallbacks = [n.strip() for n in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if n.strip()]
    names = [primary] + [n for n in fallbacks if n != primary]
    models = [ChatGoogleGenerativeAI(model=n, temperature=0, max_retries=2) for n in names]
    return FallbackModel(models, names)


def build_agent_graph(ctx: Context, llm=None, checkpointer=None,
                      max_model_calls: int = MAX_MODEL_CALLS, memory=None):
    """memory: a UserMemory instance enables Part 3 (user memory node + remember_preference tool)."""
    tools = make_tools(ctx, memory)
    llm_with_tools = (llm or get_model()).bind_tools(tools)

    def load_user_memory(state: AgentState) -> dict:
        return {"user_memory": memory.load(ctx.user_id)}

    def agent(state: AgentState) -> dict:
        system = SystemMessage(content=build_system_prompt(
            state.get("user_memory"), memory_enabled=memory is not None))
        response = invoke_with_retry(llm_with_tools, [system] + state["messages"])
        return {"messages": [response], "model_calls": state["model_calls"] + 1}

    def limit(state: AgentState) -> dict:
        # Every pending tool call needs an answer, or the next turn would be invalid.
        pending = state["messages"][-1].tool_calls
        skipped = [ToolMessage(content="Not executed: loop limit reached.",
                               tool_call_id=tc["id"], name=tc["name"]) for tc in pending]
        stop = AIMessage(content=f"I stopped after {max_model_calls} model calls without finishing "
                                 "(loop limit). Please rephrase or try again.")
        return {"messages": skipped + [stop], "limit_reached": True}

    def route(state: AgentState) -> str:
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools" if state["model_calls"] < max_model_calls else "limit"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("limit", limit)
    if memory is not None:
        builder.add_node("load_user_memory", load_user_memory)
        builder.add_edge(START, "load_user_memory")
        builder.add_edge("load_user_memory", "agent")
    else:
        builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, ["tools", "limit", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("limit", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


def text_of(content) -> str:
    """Model content can be a string or a list of parts (Gemini sometimes does this)."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        parts.append(p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else "")
    return "".join(parts)


def run_turn(graph, ctx: Context, message: str) -> dict:
    """Send one user message; return reply, step-by-step trace, information sources and flags."""
    config = {"configurable": {"thread_id": ctx.thread_id}}
    tickets_before, writes_before = len(ctx.tickets_created), len(ctx.memory_writes)
    earlier = graph.get_state(config).values.get("messages", [])  # saved conversation state
    steps, reply, loaded_memory = [], "", {}
    inp = {"messages": [HumanMessage(content=message)], "model_calls": 0, "limit_reached": False}
    for chunk in graph.stream(inp, config, stream_mode="updates"):
        for node, update in chunk.items():
            step = {"node": node}
            extra = {k: v for k, v in update.items() if k != "messages"}
            if extra:
                step["state_update"] = extra
            if node == "load_user_memory":
                loaded_memory = update.get("user_memory", {})
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        step.setdefault("model_tool_requests", []).extend(
                            {"tool": tc["name"], "args": tc["args"]} for tc in m.tool_calls)
                    else:
                        reply = text_of(m.content)
                    if text_of(m.content):
                        step["model_text"] = text_of(m.content)
                elif isinstance(m, ToolMessage):
                    step.setdefault("tool_results", []).append(
                        {"tool": m.name, "result": text_of(m.content)})
            steps.append(step)
    final = graph.get_state(config).values
    used = {r["tool"] for s in steps for r in s.get("tool_results", [])}
    return {
        "message": message, "reply": reply, "steps": steps,
        "sources": {
            "reference_file": sorted(used & {"find_instructions", "check_service_status"}),
            "conversation_state": (f"{len(earlier)} earlier messages saved for this thread_id"
                                   if earlier else "none (new thread)"),
            "user_memory_loaded": loaded_memory,
            "user_memory_written": ctx.memory_writes[writes_before:],
        },
        "tickets": ctx.tickets_created[tickets_before:],
        "model_calls": final.get("model_calls"),
        "loop_limit_reached": bool(final.get("limit_reached")),
    }
