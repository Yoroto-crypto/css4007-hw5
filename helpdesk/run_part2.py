"""Part 2 runner: scenarios S06-S09 and S14, direct tool calls, or interactive chat.

  python -m helpdesk.run_part2 --scenario S06
  python -m helpdesk.run_part2 --all
  python -m helpdesk.run_part2 --tools-demo
  python -m helpdesk.run_part2 --chat --user demo-user-a --thread t1 --profile normal
"""
import argparse
import json
import os

from helpdesk.agent import build_agent_graph, get_model, run_turn
from helpdesk.classify import DATA_DIR, ROOT
from helpdesk.tools import Context, make_tools

OUT_DIR = ROOT / "outputs"
PART2_IDS = ["S06", "S07", "S08", "S09", "S14"]


def show_turn(result: dict):
    for s in result["steps"]:
        for r in s.get("model_tool_requests", []):
            print(f"  [model -> tool] {r['tool']}({r['args']})")
        for r in s.get("tool_results", []):
            print(f"  [tool result]   {r['tool']}: {r['result'][:300]}")
    print(f"helpdesk: {result['reply']}")
    if result["tickets"]:
        print(f"  ticket file(s): {result['tickets']}")
    if result["loop_limit_reached"]:
        print("  !! loop limit reached")


def run_scenario(sid: str):
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as f:
        sc = {s["id"]: s for s in json.load(f)["scenarios"]}[sid]
    first = sc["events"][0]
    ctx = Context(user_id=first["user_id"], thread_id=first["thread_id"],
                  status_profile=sc["status_profile"])
    graph = build_agent_graph(ctx)  # new graph => new in-memory conversation (reset_before)
    print(f"\n=== {sid}: {sc['title']} | profile={ctx.status_profile} ===")
    turns = []
    for ev in sc["events"]:
        if ev["action"] != "message":
            continue
        print(f"user: {ev['message']}")
        result = run_turn(graph, ctx, ev["message"])
        show_turn(result)
        turns.append(result)
    log = {
        "scenario": sid, "title": sc["title"], "status_profile": ctx.status_profile,
        "user_id": ctx.user_id, "thread_id": ctx.thread_id,
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "provider": "Google AI Studio (Gemini)",
        "turns": turns,
        "sources": {"reference_file": "data/helpdesk.json via find_instructions; "
                                      "data/service_status.json via check_service_status",
                    "conversation_state": "in-memory checkpointer, same thread_id (Part 2)",
                    "user_memory": "not used in Part 2"},
        "ticket_files": ctx.tickets_created,
        "loop_limit_reached": any(t["loop_limit_reached"] for t in turns),
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{sid}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def tools_demo():
    """Call every tool directly, without a model or a graph."""
    ctx = Context(user_id="demo-user-a", thread_id="tools-demo", status_profile="normal")
    t = {x.name: x for x in make_tools(ctx)}
    calls = [
        ("find_instructions", {"topic": "wifi"}),
        ("find_instructions", {"topic": "printer"}),
        ("check_service_status", {"service": "wifi"}),
        ("check_service_status", {"service": "account_recovery"}),
        ("prepare_ticket", {"category": "wifi", "summary": "", "steps_tried": []}),
        ("prepare_ticket", {"category": "wifi", "summary": "Campus-Demo will not connect.",
                            "steps_tried": ["enabled Wi-Fi", "forgot saved network"],
                            "service_status": "operational"}),
    ]
    log = []
    for name, args in calls:
        result = t[name].invoke(args)
        print(f"{name}({args})\n  -> {json.dumps(result, ensure_ascii=False)[:400]}")
        log.append({"tool": name, "args": args, "result": result})
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "part2_direct_tool_calls.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def chat(user, thread, profile):
    ctx = Context(user_id=user, thread_id=thread, status_profile=profile)
    graph = build_agent_graph(ctx)
    print(f"user={user} thread={thread} profile={profile}. Empty line to quit.")
    while (msg := input("you> ").strip()):
        show_turn(run_turn(graph, ctx, msg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=PART2_IDS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--tools-demo", action="store_true")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--user", default="demo-user-a")
    p.add_argument("--thread", default="thread-1")
    p.add_argument("--profile", default="normal",
                   choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    a = p.parse_args()
    if a.tools_demo:
        tools_demo()
    elif a.chat:
        chat(a.user, a.thread, a.profile)
    elif a.all:
        for sid in PART2_IDS:
            run_scenario(sid)
    elif a.scenario:
        run_scenario(a.scenario)
    else:
        p.error("use --scenario, --all, --tools-demo or --chat")


if __name__ == "__main__":
    main()
