"""Part 1: a helpdesk graph WITHOUT an LLM.

original:            START -> read_request -> {wifi_help | password_help | other_help} -> END
with_clarification:  same, plus clarify_problem when both Wi-Fi and password are mentioned.
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from helpdesk.classify import classify, format_instructions


class HelpdeskState(TypedDict, total=False):
    message: str    # the user's original message (never changed after START)
    category: str   # set by read_request: wifi / password / other / both
    reply: str      # set by one of the help nodes


# ---------- nodes: each reads the state and returns an UPDATE (a dict) ----------

def make_read_request(detect_both: bool):
    def read_request(state: HelpdeskState) -> dict:
        return {"category": classify(state["message"], detect_both=detect_both)}
    return read_request


def wifi_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {"reply": "Is your problem about Wi-Fi or about a password?"}


def clarify_problem(state: HelpdeskState) -> dict:
    return {"reply": "Is the problem connecting to Wi-Fi, or recovering your account password? "
                     "Which one would you like to solve first?"}


# ---------- conditional edge: reads the state and returns the NEXT NODE'S NAME ----------

def route_after_read(state: HelpdeskState) -> Literal["wifi_help", "password_help",
                                                      "other_help", "clarify_problem"]:
    return {
        "wifi": "wifi_help",
        "password": "password_help",
        "both": "clarify_problem",
    }.get(state["category"], "other_help")


def build_graph(variant: str = "original"):
    if variant not in ("original", "with_clarification"):
        raise ValueError(f"unknown variant: {variant}")
    with_clarify = variant == "with_clarification"

    builder = StateGraph(HelpdeskState)
    builder.add_node("read_request", make_read_request(detect_both=with_clarify))
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    if with_clarify:
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", route_after_read, targets)
    for node in targets:
        builder.add_edge(node, END)

    return builder.compile()


def run_with_trace(graph, message: str):
    """Run the graph and record the node order and the state after every node."""
    state = {"message": message}
    steps = []
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **update}
            steps.append({"node": node, "update": update, "state_after": dict(state)})
    return state, steps
