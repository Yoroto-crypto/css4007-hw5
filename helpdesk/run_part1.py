"""Run Part 1 scenarios (S01-S05) or any message, print the trace, save a log to outputs/.

Examples (from the repo root):
  python -m helpdesk.run_part1 --scenario S01
  python -m helpdesk.run_part1 --all
  python -m helpdesk.run_part1 --message "My Wi-Fi password does not work." --variant with_clarification --name custom1
"""
import argparse
import json

from helpdesk.classify import DATA_DIR, ROOT
from helpdesk.part1_graph import build_graph, run_with_trace

OUT_DIR = ROOT / "outputs"
PART1_IDS = ["S01", "S02", "S03", "S04", "S05"]


def load_scenarios() -> dict:
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as f:
        return {s["id"]: s for s in json.load(f)["scenarios"]}


def run_one(name, message, variant, user_id, thread_id, status_profile):
    graph = build_graph(variant)
    final_state, steps = run_with_trace(graph, message)

    print(f"\n=== {name} | variant={variant} ===")
    print(f"message: {message}")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. node={s['node']:<16} update={s['update']}")
    print(f"route: START -> {' -> '.join(s['node'] for s in steps)} -> END")
    print(f"reply:\n{final_state['reply']}")

    log = {
        "scenario": name,
        "graph_variant": variant,
        "status_profile": status_profile,
        "user_id": user_id,
        "thread_id": thread_id,
        "message": message,
        "nodes_visited": [s["node"] for s in steps],
        "steps": steps,
        "sources": {"reference_file": "data/helpdesk.json (help nodes only)",
                    "conversation_state": "not used in Part 1",
                    "user_memory": "not used in Part 1"},
        "final_reply": final_state["reply"],
        "ticket_file": None,
        "loop_limit_reached": False,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                          encoding="utf-8")


def save_diagrams():
    """Save Mermaid diagrams of both graphs (paste into SUBMISSION.md)."""
    OUT_DIR.mkdir(exist_ok=True)
    for variant in ("original", "with_clarification"):
        mermaid = build_graph(variant).get_graph().draw_mermaid()
        (OUT_DIR / f"graph_{variant}.mmd").write_text(mermaid, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=PART1_IDS)
    p.add_argument("--all", action="store_true", help="run S01-S05 and save diagrams")
    p.add_argument("--message")
    p.add_argument("--variant", default="original", choices=["original", "with_clarification"])
    p.add_argument("--name", default="custom")
    args = p.parse_args()

    scenarios = load_scenarios()
    if args.all or args.scenario:
        ids = PART1_IDS if args.all else [args.scenario]
        for sid in ids:
            sc = scenarios[sid]
            ev = sc["events"][0]
            run_one(sid, ev["message"], sc.get("graph_variant", "original"),
                    ev["user_id"], ev["thread_id"], sc["status_profile"])
        if args.all:
            save_diagrams()
    elif args.message:
        run_one(args.name, args.message, args.variant, None, None, None)
    else:
        p.error("use --scenario, --all or --message")


if __name__ == "__main__":
    main()
