# HW5 — University IT helpdesk with LangGraph

A terminal helpdesk built in four stages (CSS-4007). All data is fictional (`data/`); tickets are
local drafts and nothing is ever sent. Detailed explanations and results: [SUBMISSION.md](SUBMISSION.md).

**Python:** 3.12 (check yours with `python --version`). Exact package versions: `requirements.txt`.

## 1. Install

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Configure the model (Parts 2–3 only; Part 1 needs no key)

Provider: Google AI Studio (Gemini) through `langchain-google-genai`.

1. Create an API key in Google AI Studio.
2. Copy `.env.example` to `.env` and fill it in (never commit `.env`):
   ```
   GOOGLE_API_KEY=your-key
   GEMINI_MODEL=<a tool-capable model that has free quota for your project>
   GEMINI_FALLBACK_MODELS=<optional, comma-separated backup models>
   ```
   Check which models your project can use at https://ai.dev/rate-limit.
   If the main model is overloaded (503) the program retries with waits, then tries the backups.

## 3. Run

**Part 1 — graph without a model** (S01–S05; saves logs and Mermaid diagrams to `outputs/`):
```
python -m helpdesk.run_part1 --all
python -m helpdesk.run_part1 --scenario S04
python -m helpdesk.run_part1 --message "I forgot my PASSWORD." --variant with_clarification --name custom1
```

**Part 2 — the three tools called directly, no model:**
```
python -m helpdesk.run_part2 --tools-demo
```

**Parts 2 + 3 — the full helpdesk** (S06–S14; each scenario resets `storage/` first):
```
python -m helpdesk.main --all
python -m helpdesk.main --scenario S11
```

**Your own conversation** (user id, thread id and status profile are arguments;
profiles: `normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`):
```
python -m helpdesk.main --user demo-user-a --thread t1 --profile normal --chat
python -m helpdesk.main --user demo-user-a --thread t1 --message "My issue is Wi-Fi. I enabled it and reconnecting failed."
```
Each `--message` launch is a separate process. Real restart demonstration:
```
python -m helpdesk.main --reset
python -m helpdesk.main --user demo-user-a --thread restart-a --message "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
python -m helpdesk.main --user demo-user-a --thread restart-a --message "Which problem and steps did we discuss?"
python -m helpdesk.main --user demo-user-a --thread brand-new --message "Which language did I ask you to use?"
python -m helpdesk.main --user demo-user-b --thread isolation-b --message "Have I told you a preferred language or a previous problem?"
```
Every turn prints the tool calls, tool results and the information sources
(reference file / conversation state / user memory) and is saved in `outputs/`.

## 4. Where memory lives and how to clear it

| What | Where | Key |
| --- | --- | --- |
| Conversation state (LangGraph `SqliteSaver`) | `storage/conversations.sqlite` | `thread_id` |
| User preferences (plain `sqlite3` table) | `storage/user_memory.sqlite` | `user_id` |
| Ticket drafts | `tickets/TKT-*.json` | ticket id |

Clear conversations and memory: `python -m helpdesk.main --reset` (deletes `storage/`).
`storage/` and `.env` are git-ignored. Logs in `outputs/` are committed; they use only invented ids.

## 5. Tests and CI

```
python -m pytest -q
```
No API key needed. Tests that involve a model use a labelled fake (`ScriptedModel`).
GitHub Actions (`.github/workflows/tests.yml`) runs the same command on every push.
The real model demonstration is separate (`python -m helpdesk.main --all`).

## Layout

```
helpdesk/classify.py     keyword rules + helpdesk.json access (Part 1)
helpdesk/part1_graph.py  graph without a model; run_part1.py runs S01-S05
helpdesk/tools.py        find_instructions, check_service_status, prepare_ticket, remember_preference
helpdesk/agent.py        model <-> tools loop, loop limit, user-memory node, retries/fallback
helpdesk/memory.py       per-user preference store
helpdesk/main.py         the program (scenarios, chat, reset)
tests/                   routing, tools, restart, isolation
```
