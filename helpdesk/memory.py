"""Persistent per-user memory (preferences), stored in its own SQLite file.

This is separate from the conversation state (the LangGraph checkpointer, keyed by thread_id).
Here everything is keyed by user_id, so it survives new conversations and process restarts,
and one user can never read another user's rows.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class UserMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: LangGraph may run a tool in a worker thread.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences ("
                "user_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, PRIMARY KEY (user_id, name))")
            self._conn.commit()

    def save(self, user_id: str, name: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO preferences (user_id, name, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, name) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at", (user_id, name, value, now))
            self._conn.commit()

    def load(self, user_id: str) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, value FROM preferences WHERE user_id = ? ORDER BY name",
                (user_id,)).fetchall()
        return {name: value for name, value in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
