"""The helpdesk program (Parts 2 + 3): SQLite conversation state + per-user memory.

Examples (from the repo root):
  python -m helpdesk.main --scenario S11
  python -m helpdesk.main --all                       # S06-S14
  python -m helpdesk.main --user demo-user-a --thread restart-a --message "I have a Wi-Fi problem."
  python -m helpdesk.main --user demo-user-a --thread restart-a --chat
  python -m helpdesk.main --reset                     # delete saved conversations and memory

Running --message twice with the same --user/--thread is a REAL restart: the second launch has
nothing in memory except what the SQLite files in storage/ contain.
"""
import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver

from helpdesk.agent import build_agent_graph, get_model, run_turn
from helpdesk.classify import DATA_DIR, ROOT
from helpdesk.memory import UserMemory
from helpdesk.run_part2 import show_turn
from helpdesk.tools import Context

STORAGE = ROOT / "storage"
OUT_DIR = ROOT / "outputs"
SCENARIO_IDS = [f"S{n:02d}" for n in range(6, 15)]
PROFILES = ["normal", "wifi_outage", "recovery_outage", "status_unavailable"]


class Runtime:
    """Everything one 'launch' of the program needs for one user + thread."""

    def __init__(self, user_id, thread_id, profile, llm, storage_dir=STORAGE):
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(user_id=user_id, thread_id=thread_id, status_profile=profile)
        self.memory = UserMemory(storage_dir / "user_memory.sqlite")
        self._conn = sqlite3.connect(storage_dir / "conversations.sqlite", check_same_thread=False)
        self.graph = build_agent_graph(self.ctx, llm=llm, checkpointer=SqliteSaver(self._conn),
                                       memory=self.memory)

    def ask(self, message: str) -> dict:
        return run_turn(self.graph, self.ctx, message)

    def close(self):
        self.memory.close()
        self._conn.close()


def reset_storage():
    shutil.rmtree(STORAGE, ignore_errors=True)


def show(result: dict):
    show_turn(result)
    src = result["sources"]
    print(f"  sources: reference={src['reference_file']} | conversation={src['conversation_state']} "
          f"| user memory loaded={src['user_memory_loaded']} | memory written={src['user_memory_written']}")


def run_scenario(sid: str, llm):
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as f:
        sc = {s["id"]: s for s in json.load(f)["scenarios"]}[sid]
    if sc.get("reset_before"):
        reset_storage()
    profile = sc["status_profile"]
    print(f"\n=== {sid}: {sc['title']} | profile={profile} ===")
    runtimes, events_log = {}, []
    try:
        _run_events(sc, profile, llm, runtimes, events_log)
    finally:
        tickets = [t for rt in runtimes.values() for t in rt.ctx.tickets_created]
        for rt in runtimes.values():
            rt.close()
    _write_log(sid, sc, profile, events_log, tickets)


def _run_events(sc, profile, llm, runtimes, events_log):
    for ev in sc["events"]:
        if ev["action"] == "restart_process":
            for rt in runtimes.values():
                rt.close()
            runtimes.clear()  # every object is rebuilt from the saved files on the next message
            print("--- process restarted: connections closed, only storage/ files kept ---")
            events_log.append({"action": "restart_process", "preserve_local_storage": True})
            continue
        key = (ev["user_id"], ev["thread_id"])
        if key not in runtimes:
            runtimes[key] = Runtime(ev["user_id"], ev["thread_id"], profile, llm)
        print(f"[{ev['user_id']} / {ev['thread_id']}] user: {ev['message']}")
        result = runtimes[key].ask(ev["message"])
        show(result)
        events_log.append({"action": "message", "user_id": ev["user_id"],
                           "thread_id": ev["thread_id"], **result})


def _write_log(sid, sc, profile, events_log, tickets):
    turns = [e for e in events_log if e["action"] == "message"]
    log = {
        "scenario": sid, "title": sc["title"], "status_profile": profile,
        "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "provider": "Google AI Studio (Gemini)",
        "storage": {"conversation_state": "storage/conversations.sqlite (SqliteSaver, by thread_id)",
                    "user_memory": "storage/user_memory.sqlite (by user_id)"},
        "events": events_log, "ticket_files": tickets,
        "loop_limit_reached": any(t["loop_limit_reached"] for t in turns),
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{sid}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def save_adhoc_log(user, thread, profile, result):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"chat_{user}_{thread}.json"
    turns = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    turns.append({"time": datetime.now().isoformat(timespec="seconds"), "user_id": user,
                  "thread_id": thread, "status_profile": profile, **result})
    path.write_text(json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIO_IDS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--user", default="demo-user-a")
    p.add_argument("--thread", default="thread-1")
    p.add_argument("--profile", default="normal", choices=PROFILES)
    p.add_argument("--message")
    p.add_argument("--chat", action="store_true")
    a = p.parse_args()

    if a.reset:
        reset_storage()
        print("storage/ deleted: saved conversations and user memory are gone.")
    elif a.scenario or a.all:
        llm = get_model()
        failed = []
        for sid in (SCENARIO_IDS if a.all else [a.scenario]):
            try:
                run_scenario(sid, llm)
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                print(f"\n!! {sid} FAILED: {type(exc).__name__}: {str(exc)[:300]}")
                failed.append(sid)
        if failed:
            print(f"\nFailed scenarios: {failed}. Rerun each with: python -m helpdesk.main --scenario <ID>")
            raise SystemExit(1)
    elif a.message or a.chat:
        rt = Runtime(a.user, a.thread, a.profile, get_model())
        print(f"user={a.user} thread={a.thread} profile={a.profile}")
        try:
            if a.message:
                result = rt.ask(a.message)
                show(result)
                save_adhoc_log(a.user, a.thread, a.profile, result)
            else:
                print("Empty line to quit.")
                while msg := input("you> ").strip():
                    result = rt.ask(msg)
                    show(result)
                    save_adhoc_log(a.user, a.thread, a.profile, result)
        finally:
            rt.close()
    else:
        p.error("use --scenario, --all, --message, --chat or --reset")


if __name__ == "__main__":
    main()
